const functions = require("firebase-functions");
const admin = require("firebase-admin");
const axios = require("axios");

admin.initializeApp();

// --- 환경 변수 ---
const SERVICE_KEY = functions.config().police.service_key;
const KAKAO_API_KEY = functions.config().kakao.key; // 🔑 카카오맵 API 키
const API_URL_police = functions.config().police.url_police;
const API_URL_portal = functions.config().police.url_portal;

// --- 함수 실행 옵션 ---
const runtimeOpts = {
  timeoutSeconds: 540,
  memory: "512MB",
};

// --- 잠시 기다리는 함수 (재시도를 위해) ---
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// --- YYYYMMDD 형식 날짜 변환 함수 ---
const getFormattedDate = (date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}${month}${day}`;
};

// --- 카카오맵 API 호출을 위한 공통 함수 ---
const getCoordinates = async (query, docId, collectionName) => {
  if (!query) return null;
  const kakaoApiUrl = `https://dapi.kakao.com/v2/local/search/keyword.json`;
  // 👇 [수정됨] 최대 재시도 횟수를 4회로 늘렸습니다.
  const MAX_RETRIES = 4;
  for (let i = 0; i < MAX_RETRIES; i++) {
    try {
      const kakaoResponse = await axios.get(kakaoApiUrl, {
        params: { query: query },
        headers: { Authorization: `KakaoAK ${KAKAO_API_KEY}` },
        timeout: 11000,
      });
      const documents = kakaoResponse.data.documents;
      if (documents && documents.length > 0) return documents[0]; // 성공 시 첫 번째 결과 반환
      return null;
    } catch (error) {
      if (error.response && error.response.status === 429) {
        // 👇 [수정됨] 재시도 대기 시간을 1.5배로 늘렸습니다. (1.5초, 3초, 6초, 12초)
        const waitTime = Math.pow(2, i) * 1500;
        console.warn(`[${collectionName} 좌표] ${docId} API 사용량 초과(429). ${waitTime}ms 후 재시도...`);
        await delay(waitTime);
      } else {
        console.error(`[${collectionName} 좌표] ${docId} 카카오맵 오류:`, error.message);
        return null;
      }
    }
  }
  console.error(`[${collectionName} 좌표] ${docId} 최대 재시도 횟수(${MAX_RETRIES})를 초과했습니다.`);
  return null;
};


// ===================================================================
//   1. 모든 데이터 처리를 위한 공통 함수
// ===================================================================
const collectAndProcessData = async (apiUrl, collectionName, logPrefix) => {
  // 1-1. KST 기준 '어제' 날짜 계산
  const today = new Date(new Date().getTime() + (9 * 60 * 60 * 1000));
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  const yesterdayStr = getFormattedDate(yesterday);
  console.log(`ℹ️ [${logPrefix}] 어제 날짜(${yesterdayStr}) 데이터 수집을 시작합니다.`);

  // 1-2. API 호출
  const response = await axios.get(apiUrl, {
    params: {
      serviceKey: SERVICE_KEY,
      START_YMD: yesterdayStr,
      END_YMD: yesterdayStr,
      numOfRows: "2000",
      _type: "json"
    },
    timeout: 120000, responseType: 'json',
  });
  const items = response.data.response?.body?.items?.item || [];
  if (items.length === 0) {
    console.log(`[${logPrefix}] API에서 어제 날짜 데이터가 없습니다.`);
    return `[${logPrefix}] API에서 어제 날짜 데이터가 없습니다.`;
  }
  console.log(`✅ [${logPrefix}] API 처리 완료: ${items.length}건`);

  // 1-3. 좌표 검색 및 주소 문자열 필터링
  console.log(`ℹ️ [${logPrefix}] ${items.length}건에 대한 좌표 검색 및 필터링 시작...`);
  const enrichedItems = await Promise.all(items.map(async (item) => {
    const place = item.depPlace;
    const kakaoResult = await getCoordinates(place, item.atcId, logPrefix);
    
    if (kakaoResult) {
      const addressString = kakaoResult.road_address_name || kakaoResult.address_name || "";
      if (addressString.includes("서울") || addressString.includes("수원")) {
        const lat = parseFloat(kakaoResult.y);
        const lng = parseFloat(kakaoResult.x);
        return {
          ...item,
          location: new admin.firestore.GeoPoint(lat, lng),
          addr: addressString,
        };
      }
    }
    return null; // 주소가 없거나, 주소에 서울/수원이 포함되지 않으면 null 반환
  }));

  const finalItemsToSave = enrichedItems.filter(item => item !== null);
  if (finalItemsToSave.length === 0) {
    console.log(`[${logPrefix}] 필터링 후 저장할 데이터가 없습니다.`);
    return `[${logPrefix}] 필터링 후 저장할 데이터가 없습니다.`;
  }
  console.log(`✅ [${logPrefix}] 주소 필터링 완료: ${finalItemsToSave.length}건`);

  // 1-4. Firestore에 저장
  const db = admin.firestore();
  const batch = db.batch();
  const collectionRef = db.collection(collectionName);
  finalItemsToSave.forEach(item => {
    const docRef = collectionRef.doc(item.atcId);
    const processedItem = {
      atcId: item.atcId || null, itemName: item.fdPrdtNm || null,
      itemCategory: item.prdtClNm || null, foundDate: item.fdYmd || null,
      storagePlace: item.depPlace || null,
      addr: item.addr || null,
      location: item.location || null,
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
      imageUrl: item.fdFilePathImg || null
    };
    batch.set(docRef, processedItem);
  });
  await batch.commit();

  return `✅ 성공! [${logPrefix}] 최종 ${finalItemsToSave.length}건의 데이터를 Firestore에 저장했습니다.`;
};


// ===================================================================
//   2. 각 API를 호출하는 HTTP 트리거 함수들
// ===================================================================
exports.collectPoliceData = functions
  .region("asia-northeast3")
  .runWith(runtimeOpts)
  .https.onRequest(async (req, res) => {
    try {
      const message = await collectAndProcessData(API_URL_police, "testPoliceLostItem", "경찰청");
      console.log(message);
      res.status(200).send(message);
    } catch (error) {
      console.error("🚨 경찰청 함수 실행 중 오류:", error.message);
      res.status(500).send("오류 발생");
    }
  });

exports.collectPortalData = functions
  .region("asia-northeast3")
  .runWith(runtimeOpts)
  .https.onRequest(async (req, res) => {
    try {
      const message = await collectAndProcessData(API_URL_portal, "testPortalLostItem", "포털기관");
      console.log(message);
      res.status(200).send(message);
    } catch (error) {
      console.error("🚨 포털기관 함수 실행 중 오류:", error.message);
      res.status(500).send("오류 발생");
    }
  });
