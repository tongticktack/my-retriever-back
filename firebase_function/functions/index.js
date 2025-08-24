const functions = require("firebase-functions");
const admin = require("firebase-admin");
const axios = require("axios");

admin.initializeApp();

const SERVICE_KEY = functions.config().police.service_key; //공공데이터 포털 api 서비스 키
const KAKAO_API_KEY = functions.config().kakao.key; // 카카오맵 api 키
const API_URL_police = functions.config().police.url_police; // 경찰청 api url
const API_URL_portal = functions.config().police.url_portal; // 포털기관 api url

const runtimeOpts = {
  timeoutSeconds: 540,
  memory: "512MB",
}; // 런타임 옵션(9분, 512MB)

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms)); // 재시도 대기 시간 함수


const getFormattedDate = (date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}${month}${day}`;
};
const today = new Date(new Date().getTime() + (9 * 60 * 60 * 1000));
const yesterday = new Date(today);
yesterday.setDate(today.getDate() - 1);
const yesterdayStr = getFormattedDate(yesterday);
// daily 호출용 YYYYMMDD 형식 날짜 변환 함수

// 카카오맵 API 호출을 위한 공통 함수
// 경찰청/포털기관 선택
const getCoordinates = async (query, docId, collectionName) => {
  if (!query) return null;
  const kakaoApiUrl = `https://dapi.kakao.com/v2/local/search/keyword.json`;
  // 최대 재시도 4번
  const MAX_RETRIES = 4;
  for (let i = 0; i < MAX_RETRIES; i++) {
    try {
      const kakaoResponse = await axios.get(kakaoApiUrl, {
        params: { query: query },
        headers: { Authorization: `KakaoAK ${KAKAO_API_KEY}` },
        timeout: 13000,
      });
      const documents = kakaoResponse.data.documents;
      if (documents && documents.length > 0) return documents[0]; // 검색 성공 시 첫 번째 결과 좌표
      return null;
    } catch (error) {
      if (error.response && error.response.status === 429) { // API 사용량 초과 시 재시도
        const waitTime = Math.pow(2, i) * 1500;
        console.warn(`[${collectionName} 좌표] ${docId} API 사용량 초과(429). ${waitTime}ms 후 재시도...`);
        await delay(waitTime);
      } else {
        console.error(`[${collectionName} 좌표] ${docId} 카카오맵 오류:`, error.message);
        return null;
      }
    }
  }
  console.error(`[${collectionName} 좌표] ${docId} 최대 재시도 횟수(${MAX_RETRIES})를 초과`);
  return null;
};

// 데이터 수집 가공 기록
const collectAndProcessData = async (apiUrl, collectionName, logPrefix) => {

  // api 호출
  const response = await axios.get(apiUrl, {
    params: {
      serviceKey: SERVICE_KEY,
      START_YMD: yesterdayStr, // 어제 날짜
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
  } // 데이터 없을 경우 종료
  console.log(`[${logPrefix}] API 처리 완료: ${items.length}건`);

  // 주소 & 좌표 검색 및 필터링
  const enrichedItems = await Promise.all(items.map(async (item) => {
    const place = item.depPlace; //보관 장소 기준 검색
    const kakaoResult = await getCoordinates(place, item.atcId, logPrefix); //좌표 & 주소 검색

    if (kakaoResult) {
      const addressString = kakaoResult.road_address_name || kakaoResult.address_name || "";
      if (addressString.includes("서울") || addressString.includes("수원")) { // 서울/수원 주소 필터링
        const lat = parseFloat(kakaoResult.y);
        const lng = parseFloat(kakaoResult.x);
        return {
          ...item,
          location: new admin.firestore.GeoPoint(lat, lng), // Firestore GeoPoint 좌표 추가
          addr: addressString,  // 주소 정보 추가
        };
      }
    }
    return null; // 주소가 없거나, 주소에 서울/수원이 포함되지 않으면 null 반환
  }));

  const finalItemsToSave = enrichedItems.filter(item => item !== null);
  if (finalItemsToSave.length === 0) {
    console.log(`[${logPrefix}] 필터링 후 저장할 데이터가 없습니다.`);
    return `[${logPrefix}] 필터링 후 저장할 데이터가 없습니다.`;
  } // 필터링 후 데이터가 없으면 종료
  console.log(`[${logPrefix}] 주소 필터링 완료: ${finalItemsToSave.length}건`);

  // Firestore 저장
  const db = admin.firestore();
  const batch = db.batch();
  const collectionRef = db.collection(collectionName); // 경찰청/포털기관 컬렉션 저장
  finalItemsToSave.forEach(item => {
    const docRef = collectionRef.doc(item.atcId);
    const processedItem = {
      atcId: item.atcId || null, // 고유 식별 번호
      itemName: item.fdPrdtNm || null, // 물품명
      itemCategory: item.prdtClNm || null, // 물품 카테고리
      foundDate: item.fdYmd || null, // 습득 일자
      storagePlace: item.depPlace || null, // 보관 장소
      addr: item.addr || null, // 주소
      location: item.location || null, // 좌표
      createdAt: admin.firestore.FieldValue.serverTimestamp(), // 생성 일시
      imageUrl: item.fdFilePathImg || null  // 이미지 URL
    };
    batch.set(docRef, processedItem);
  });
  await batch.commit();

  return `[${logPrefix}] 최종 ${finalItemsToSave.length}건의 데이터를 Firestore에 저장`;
};

// 기관별 api 호출 및 데이터 수집 함수
exports.collectPoliceData = functions
  .region("asia-northeast3")
  .runWith(runtimeOpts)
  .https.onRequest(async (req, res) => {
    try {
      const message = await collectAndProcessData(API_URL_police, "PoliceLostItem", "경찰청");
      console.log(message);
      res.status(200).send(message);
    } catch (error) {
      console.error("경찰청 함수 실행 중 오류:", error.message);
      res.status(500).send("오류 발생");
    }
  });

exports.collectPortalData = functions
  .region("asia-northeast3")
  .runWith(runtimeOpts)
  .https.onRequest(async (req, res) => {
    try {
      const message = await collectAndProcessData(API_URL_portal, "PortalLostItem", "포털기관");
      console.log(message);
      res.status(200).send(message);
    } catch (error) {
      console.error("포털기관 함수 실행 중 오류:", error.message);
      res.status(500).send("오류 발생");
    }
  });