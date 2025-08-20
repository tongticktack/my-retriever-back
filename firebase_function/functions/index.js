const functions = require("firebase-functions");
const admin = require("firebase-admin");
const axios = require("axios");

admin.initializeApp();

const SERVICE_KEY = functions.config().police.service_key;//서비스 키
const API_URL_police = functions.config().police.url_police;//경찰청 지역 파라미터 기준 보관 분실물 엔드포인트
const API_URL_portal = functions.config().police.url_portal;//포털기관 지역 파라미터 기준 보관 분실물 엔드포인트 
const runtimeOpts = {
  timeoutSeconds: 540,
  memory: "512MB",
}; //서버 응답 대기 시간 9분

const getFormattedDate = (date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}${month}${day}`;
}; //YYYYMMDD 형식으로 날짜를 입력받기 때문에 스케줄러 기능을 위한 날짜 변환 형식

const today = new Date(new Date().getTime() + (9 * 60 * 60 * 1000));
const yesterday = new Date(today);
yesterday.setDate(today.getDate() - 1);//ms로 현재 시각 받아와서 날짜 객체 생성 및 어제로 시간 변경

const yesterdayStr = getFormattedDate(yesterday); //기존에 만든 날짜 -> YYYYMMDD 형식 변환

exports.collectPoliceData = functions
  .region("asia-northeast3")
  .runWith(runtimeOpts)
  .https.onRequest(async (req, res) => {
    try {
      const combinedItems = [];

      const processApiCall = async (apiUrl, apiName) => {
        try {
          console.log(`ℹ️ [START] ${apiName} API 호출을 시작`);

          const response = await axios.get(apiUrl, {
            params: {
              serviceKey: SERVICE_KEY,
              pageNo: "1",
              END_YMD: "20250820",
              N_FD_LCT_CD: "LCA000", // 지역구
              numOfRows: "2000", //데일리로 전날 n개 데이터 db화
              _type: "json"
            },
            timeout: 120000,//타임아웃 2분
            responseType: 'json', //처음 들어오는 문자가 "<"이 아닌 "{"로 xml타입이 아니라 json 파일로 보내주는 것을 확인하고 응답 타입 변경함
          }); //파라미터 전달

          const result = response.data;
          const responseNode = result.response;

          if (!responseNode || responseNode.header.resultCode !== '00') {
            const errorHeader = responseNode?.header;
            const errorCode = errorHeader?.resultCode || 'N/A';
            const errorMsg = errorHeader?.resultMsg || '알 수 없는 오류';
            console.error(`🚨 ${apiName} API 오류 반환. 코드: ${errorCode}, 메시지: ${errorMsg}`);
            return;
          }//api에서 오류 검출 시 반환

          const items = responseNode.body?.items?.item || [];//태그에 공백 없을 경우 push
          combinedItems.push(...items);
          console.log(`✅ ${apiName} API 처리 완료: ${items.length}건`);

        } catch (apiError) {
          console.error(`🚨 ${apiName} API 처리 중 심각한 오류 발생:`, apiError.message);
        }
      };//api 호출 및 push 함수

      await processApiCall(API_URL_police, "경찰청");

      if (combinedItems.length === 0) {
        return res.status(200).send("모든 API에서 저장할 새로운 데이터가 없음. 로그 확인 필요.");
      }

      const yesterdaysItems = combinedItems.filter(item => {
        if (typeof item.fdYmd === 'string' && item.fdYmd.includes('-')) {
          const apiDate = item.fdYmd.split('-').join('');
          return apiDate === yesterdayStr;
        }
        return false;
      });

      if (yesterdaysItems.length === 0) {
        return res.status(200).send("어제 날짜에 해당하는 새로운 데이터가 없습니다.");
      }
      console.log(`✅ 어제 날짜 데이터 필터링 완료: ${yesterdaysItems.length}건`);

      const finalItemList = Array.from(combinedItems.values());

      const db = admin.firestore();
      const batch = db.batch();
      const collectionRef = db.collection("PoliceLostItem"); //PoliceLostItem 컬렉션에 저장

      finalItemList.forEach(item => {
        const docRef = collectionRef.doc(item.atcId);

        const processedItem = {
          atcId: item.atcId || null, //경찰청 식별 번호
          itemName: item.fdPrdtNm || null, //이름
          itemCategory: item.prdtClNm || null, //카테고리 대분류 > 상세분류
          foundDate: item.fdYmd || null, //습득일자
          storagePlace: item.depPlace || null, //보관장소
          location: null, //포털기관 좌표 받아서 저장할 예정
          createdAt: admin.firestore.FieldValue.serverTimestamp(),//db화된 작업 시간 확인
          imageUrl: item.fdFilePathImg || null
        };

        batch.set(docRef, processedItem);
      });

      await batch.commit();

      const successMessage = `✅ 성공 최종 ${finalItemList.length}건 데이터 Firestore에 저장`;
      console.log(successMessage);
      res.status(200).send(successMessage);

    } catch (error) {
      console.error("🚨 함수 전체 실행 중 심각한 오류 발생:", error);
      res.status(500).send(`🚨 함수 전체 실행 중 심각한 오류 발생: ${error.message}`);
    }
  }); //경찰청 api 호출 URL

exports.collectPortalData = functions
  .region("asia-northeast3")
  .runWith(runtimeOpts)
  .https.onRequest(async (req, res) => {
    try {
      const combinedItems = [];

      const processApiCall = async (apiUrl, apiName) => {
        try {
          console.log(`ℹ️ [START] ${apiName} API 호출을 시작`);

          const response = await axios.get(apiUrl, {
            params: {
              serviceKey: SERVICE_KEY,
              pageNo: "1",
              END_YMD: "20250820",
              N_FD_LCT_CD: "LCA000", // 지역구
              numOfRows: "2000", //데일리로 전날 n개 데이터 db화
              _type: "json"
            },
            timeout: 120000,//타임아웃 2분
            responseType: 'json', //처음 들어오는 문자가 "<"이 아닌 "{"로 xml타입이 아니라 json 파일로 보내주는 것을 확인하고 응답 타입 변경함
          }); //파라미터 전달

          const result = response.data;
          const responseNode = result.response;

          if (!responseNode || responseNode.header.resultCode !== '00') {
            const errorHeader = responseNode?.header;
            const errorCode = errorHeader?.resultCode || 'N/A';
            const errorMsg = errorHeader?.resultMsg || '알 수 없는 오류';
            console.error(`🚨 ${apiName} API 오류 반환. 코드: ${errorCode}, 메시지: ${errorMsg}`);
            return;
          }//api에서 오류 검출 시 반환

          const items = responseNode.body?.items?.item || [];//태그에 공백 없을 경우 push
          combinedItems.push(...items);
          console.log(`✅ ${apiName} API 처리 완료: ${items.length}건`);

        } catch (apiError) {
          console.error(`🚨 ${apiName} API 처리 중 심각한 오류 발생:`, apiError.message);
        }
      };//api 호출 및 push 함수

      await processApiCall(API_URL_portal, "포털기관");//api 분리 호출

      if (combinedItems.length === 0) {
        return res.status(200).send("모든 API에서 저장할 새로운 데이터가 없음. 로그 확인 필요.");
      }

      // const yesterdaysItems = combinedItems.filter(item => {
      //   if (typeof item.fdYmd === 'string' && item.fdYmd.includes('-')) {
      //     const apiDate = item.fdYmd.split('-').join('');
      //     return apiDate === yesterdayStr;
      //   }
      //   return false;
      // });

      // if (yesterdaysItems.length === 0) {
      //   return res.status(200).send("어제 날짜에 해당하는 새로운 데이터가 없습니다.");
      // }
      // console.log(`✅ 어제 날짜 데이터 필터링 완료: ${yesterdaysItems.length}건`);

      const finalItemList = Array.from(combinedItems.values());

      const db = admin.firestore();
      const batch = db.batch();
      const collectionRef = db.collection("PortalLostItem"); //testPortalLostItem2 컬렉션에 저장

      finalItemList.forEach(item => {
        const docRef = collectionRef.doc(item.atcId);

        const processedItem = {
          atcId: item.atcId || null, //경찰청 식별 번호
          itemName: item.fdPrdtNm || null, //이름
          itemCategory: item.prdtClNm || null, //카테고리 대분류 > 상세분류
          foundDate: item.fdYmd || null, //습득일자
          storagePlace: item.depPlace || null, //보관장소
          location: null, //포털기관 좌표 받아서 저장할 예정
          createdAt: admin.firestore.FieldValue.serverTimestamp(),//db화된 작업 시간 확인
          imageUrl: item.fdFilePathImg || null
        };

        batch.set(docRef, processedItem);
      });

      await batch.commit();

      const successMessage = `✅ 성공 최종 ${finalItemList.length}건 데이터 Firestore에 저장`;
      console.log(successMessage);
      res.status(200).send(successMessage);

    } catch (error) {
      console.error("🚨 함수 전체 실행 중 심각한 오류 발생:", error);
      res.status(500).send(`🚨 함수 전체 실행 중 심각한 오류 발생: ${error.message}`);
    }
  }); // 포털기관 api 호출 URL

const KAKAO_APP_KEY = functions.config().kakao.key;

exports.addPoliceLocations = functions
  .region("asia-northeast3")
  // .runWith({ ...runtimeOpts, concurrency: 10 })
  .firestore.document("PoliceLostItem/{docId}")
  .onCreate(async (snap, context) => {
    const newData = snap.data();
    const storagePlace = newData.storagePlace; // 보관장소 주소
    const docId = context.params.docId; // 문서 ID

    if (!storagePlace) {
      console.log(`[${docId}] 보관 주소(storagePlace) 정보가 없어 좌표 검색을 건너뜁니다.`);
      return null;
    }

    console.log(`[${docId}] '${storagePlace}'의 좌표 검색을 시작합니다...`);
    const kakaoApiUrl = "https://dapi.kakao.com/v2/local/search/keyword.json";

    try {
      const response = await axios.get(kakaoApiUrl, {
        params: {
          query: storagePlace, // 검색할 키워드 (습득 주소)
        },
        headers: {
          Authorization: `KakaoAK ${KAKAO_APP_KEY}`, // REST API 키 인증
        },
        timeout: 5000, // 5초 타임아웃
      });

      const documents = response.data.documents;

      // 4. 검색 결과가 없으면 함수를 종료합니다.
      if (!documents || documents.length === 0) {
        console.log(`[${docId}] '${storagePlace}'에 대한 좌표 검색 결과가 없습니다.`);
        return null;
      }

      // 5. 검색 결과 중 첫 번째 장소의 좌표를 사용합니다.
      const firstResult = documents[0];
      const location = {
        lat: parseFloat(firstResult.y), // 위도 (Latitude)
        lng: parseFloat(firstResult.x), // 경도 (Longitude)
      };

      console.log(`[${docId}] 좌표 검색 성공:`, location);

      // 6. 원래 문서에 'location' 필드를 추가하여 업데이트합니다.
      return snap.ref.update({
        location: new admin.firestore.GeoPoint(location.lat, location.lng)
      });

    } catch (error) {
      console.error(`[${docId}] 카카오맵 API 호출 중 오류 발생:`, error.message);
      return null;
    }
  });

exports.addPortalLocations = functions
  .region("asia-northeast3")
  // .runWith({ ...runtimeOpts, concurrency: 10 })
  .firestore.document("PortalLostItem/{docId}")
  .onCreate(async (snap, context) => {
    const newData = snap.data();
    const storagePlace = newData.storagePlace; // 보관장소 주소
    const docId = context.params.docId; // 문서 ID

    if (!storagePlace) {
      console.log(`[${docId}] 보관 주소(storagePlace) 정보가 없어 좌표 검색을 건너뜁니다.`);
      return null;
    }

    console.log(`[${docId}] '${storagePlace}'의 좌표 검색을 시작합니다...`);
    const kakaoApiUrl = "https://dapi.kakao.com/v2/local/search/keyword.json";

    try {
      const response = await axios.get(kakaoApiUrl, {
        params: {
          query: storagePlace, // 검색할 키워드 (습득 주소)
        },
        headers: {
          Authorization: `KakaoAK ${KAKAO_APP_KEY}`, // REST API 키 인증
        },
        timeout: 5000, // 5초 타임아웃
      });

      const documents = response.data.documents;

      // 4. 검색 결과가 없으면 함수를 종료합니다.
      if (!documents || documents.length === 0) {
        console.log(`[${docId}] '${storagePlace}'에 대한 좌표 검색 결과가 없습니다.`);
        return null;
      }

      // 5. 검색 결과 중 첫 번째 장소의 좌표를 사용합니다.
      const firstResult = documents[0];
      const location = {
        lat: parseFloat(firstResult.y), // 위도 (Latitude)
        lng: parseFloat(firstResult.x), // 경도 (Longitude)
      };

      console.log(`[${docId}] 좌표 검색 성공:`, location);

      // 6. 원래 문서에 'location' 필드를 추가하여 업데이트합니다.
      return snap.ref.update({
        location: new admin.firestore.GeoPoint(location.lat, location.lng)
      });

    } catch (error) {
      console.error(`[${docId}] 카카오맵 API 호출 중 오류 발생:`, error.message);
      return null;
    }
  });