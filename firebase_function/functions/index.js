const functions = require("firebase-functions");
const admin = require("firebase-admin");
const axios = require("axios");

admin.initializeApp();

const SERVICE_KEY = functions.config().police.service_key;//서비스 키
const API_URL_police = functions.config().police.url_police;//경찰청 보관 분실물 엔드포인트
const API_URL_portal = functions.config().police.url_portal;//포털기관 보관 분실물 엔드포인트

const runtimeOpts = {
  timeoutSeconds: 180,
  memory: "256MB",
}; //서버 응답 대기 시간 3분

// const getFormattedDate = (date) => {
//   const year = date.getFullYear();
//   const month = String(date.getMonth() + 1).padStart(2, '0');
//   const day = String(date.getDate()).padStart(2, '0');
//   return `${year}${month}${day}`;
// }; //YYYYMMDD 형식으로 날짜를 입력받기 때문에 스케줄러 기능을 위한 날짜 변환 형식

exports.collectPoliceData = functions
  .region("asia-northeast3")
  .runWith(runtimeOpts)
  .https.onRequest(async (req, res) => {
    try {
    //   let startDate = req.query.startDate;
    //   let endDate = req.query.endDate;
      
    //   if (!startDate || !endDate) {
    //     const today = new Date(new Date().getTime() * 60 * 60 * 1000); //ms로 현재 시각 받아와서 날짜 객체 생성
    //     const todayStr = getFormattedDate(today); //기존에 만든 날짜 -> YYYYMMDD 형식 변환
    //     startDate = todayStr;
    //     endDate = todayStr;
    //     console.log(`날짜 파라미터가 없습니다. 오늘 날짜(${todayStr})로 설정합니다.`);
    //   } else {
    //     console.log(`지정된 날짜(${startDate} ~ ${endDate})로 데이터를 검색합니다.`);
    //   }

      const combinedItems = [];
      
      const processApiCall = async (apiUrl, apiName) => {
        try {
          console.log(`ℹ️ [START] ${apiName} API 호출을 시작합니다...`);
          
          const response = await axios.get(apiUrl, { 
              params: { serviceKey: SERVICE_KEY, endDate: "20250801", numOfRows: "1000", _type: "json" }, //25년 7월부터 포털 1000개, 경찰청 1000개 데이터 받아와서 db화
              timeout: 60000,//타임아웃 1분
              responseType: 'json', //처음 들어오는 문자가 "<"이 아닌 "{"로 xml타입이 아니라 json 파일로 보내주는 것을 확인하고 응답 타입 변경함
          });
          
          const result = response.data;
          const responseNode = result.response;

          if (!responseNode || responseNode.header.resultCode !== '00') {
            const errorHeader = responseNode?.header;
            const errorCode = errorHeader?.resultCode || 'N/A';
            const errorMsg = errorHeader?.resultMsg || '알 수 없는 오류';
            console.error(`🚨 ${apiName} API가 오류를 반환했습니다. 코드: ${errorCode}, 메시지: ${errorMsg}`);
            return;
          }//api에서 오류 검출 시 반환

          const items = responseNode.body?.items?.item || [];//태그에 공백 없을 경우 push
          combinedItems.push(...items);
          console.log(`✅ ${apiName} API 처리 완료: ${items.length}건`);

        } catch (apiError) {
            console.error(`🚨 ${apiName} API 처리 중 심각한 오류 발생:`, apiError.message);
        }
      };//api 호출 및 push 함수

      await processApiCall(API_URL_police, "첫 번째 (Losfund)");
      await processApiCall(API_URL_portal, "두 번째 (LosPtfund)");//api 분리 호출

      if (combinedItems.length === 0) {
        return res.status(200).send("모든 API에서 저장할 새로운 데이터가 없습니다. 로그를 확인하세요.");
      }
      
      const uniqueItems = new Map();
      combinedItems.forEach(item => {
        if (item.atcId) uniqueItems.set(item.atcId, item);
      });
      const finalItemList = Array.from(uniqueItems.values());
      
      const db = admin.firestore();
      const batch = db.batch();
      const collectionRef = db.collection("PoliceLostItem"); //해당 컬렉션에 저장
      
      finalItemList.forEach(item => {
        const docRef = collectionRef.doc(item.atcId);
        
        const processedItem = {
          atcId: item.atcId || null, //경찰청 식별 번호
          itemName: item.fdPrdtNm || null, //이름
          itemCategory: item.prdtClNm || null, //카테고리 대분류 > 상세분류
          foundDate: item.fdYmd || null, //습득일자
          foundPlace:null, //습득장소는 null처리, 이후 크롤링 기능 개발 시 해당 필드에 삽입
          storagePlace: item.depPlace || null, //보관장소
          createdAt: admin.firestore.FieldValue.serverTimestamp(),//db화된 작업 시간 확인
        };

        const imageUrl = item.fdFilePathImg;
        processedItem.imageUrl = (imageUrl && imageUrl !== "https://www.lost112.go.kr/lostnfs/images/sub/img02_no_img.gif")? imageUrl : null; //이미지 없을 경우 null 처리

        batch.set(docRef, processedItem);
      });
      
      await batch.commit();
      
      const successMessage = `✅ 성공! 최종적으로 ${finalItemList.length}건의 데이터를 Firestore에 저장했습니다.`;
      console.log(successMessage);
      res.status(200).send(successMessage);
      
    } catch (error) {
      console.error("🚨 함수 전체 실행 중 심각한 오류 발생:", error);
      res.status(500).send(`🚨 함수 전체 실행 중 심각한 오류 발생: ${error.message}`);
    }
  });
