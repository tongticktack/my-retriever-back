import requests
import ssl
import xmltodict
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from firebase_functions import https_fn, options
from firebase_admin import initialize_app, firestore

# --- Custom SSL/TLS Adapter ---
# 이 클래스는 requests 라이브러리가 특정 보안 프로토콜(TLSv1.2)을 사용하도록 강제합니다.
# 공공 데이터 서버와의 호환성 문제를 해결하기 위한 핵심 로직입니다.
class TlsV12HttpAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLSv1_2,
        )

initialize_app()
options.set_global_options(region=options.SupportedRegion.ASIA_NORTHEAST3)

SERVICE_KEY = "s1ugtEfrXgZo9pGyoBSCkiuvPq7bEO0dlQzBeiMqpUWTOfLRx6xQgnlUA0/sGiypbE6Pnu8tmnCF+AzrvYgpEA=="

API_URL_police = "https://apis.data.go.kr/1320000/LosfundInfoInqireService" # 경찰청 습득물 api
API_URL_portal = "https://apis.data.go.kr/1320000/LosPtfundInfoInqireService" # 포털기관 습득물 api

@https_fn.on_request()
def fetch_and_store_found_data(req: https_fn.Request) -> https_fn.Response:
    try:
        combined_items = []

        # 경찰청 api 호출
        print("1/4 경찰청 습득물 api 호출 시작")
        params_police = {"serviceKey": SERVICE_KEY, "pageNo": "1", "numOfRows": "10"}
        response_police = requests.get(API_URL_police, params=params_police, timeout=10)
        response_police.raise_for_status()
        
        items_police = xmltodict.parse(response_police.content).get("response", {}).get("body", {}).get("items", {}).get("item", [])
        if items_police and not isinstance(items_police, list): items_police = [items_police]
        if items_police: combined_items.extend(items_police)
        print(f"경찰청 api 처리 완료: {len(items_police) if items_police else 0}건")

        # 포털기관 api 호출
        print("2/4 포털기관 습득물 api 호출 시작")
        params_portal = {"serviceKey": SERVICE_KEY, "pageNo": "1", "numOfRows": "10"}
        response_portal = requests.get(API_URL_portal, params=params_portal, timeout=10)
        response_portal.raise_for_status()

        items_portal = xmltodict.parse(response_portal.content).get("response", {}).get("body", {}).get("items", {}).get("item", [])
        if items_portal and not isinstance(items_portal, list): items_portal = [items_portal]
        if items_portal: combined_items.extend(items_portal)
        print(f"포털 api 처리 완료: {len(items_portal) if items_portal else 0}건")

        # 데이터 통합, 중복 제거
        if not combined_items:
            return https_fn.Response("새로운 데이터 없음", status=200)

        print(f"3/4 총 {len(combined_items)}건의 데이터 통합 및 중복 제거를 시작합...")
        unique_items = {}
        for item in combined_items:
            atc_id = item.get("atcId")
            if atc_id:
                # 딕셔너리의 키(atcId)를 사용해 중복된 항목을 자동으로 덮어쓰기
                unique_items[atc_id] = item
        
        final_item_list = list(unique_items.values())
        print(f"중복 제거 후 최종 데이터: {len(final_item_list)}건")

        # 최종 데이터를 Firestore에 저장
        print(f"4/4 총 {len(final_item_list)}건의 데이터를 Firestore에 저장")
        db = firestore.client()
        batch = db.batch()
        collection_ref = db.collection("PoliceLostItem") # 습득물 데이터 저장 컬렉션
        
        for item in final_item_list:
            doc_id = item.get("atcId")
            doc_ref = collection_ref.document(doc_id)

            processed_item = {
                "atcId": item.get("atcId"),
                "itemName": item.get("fdPrdtNm"),
                "itemCategory": item.get("prdtClNm"),
                "foundDate": item.get("fdYmd"),
                "foundPlace": item.get("fdPlace"),
                "storagePlace": item.get("depPlace"),
                "status": item.get("csteSteNm"),
                "createdAt": firestore.SERVER_TIMESTAMP,
            }
            image_url = item.get("fdFilePathImg")

            if image_url and image_url != "https://www.lost112.go.kr/lostnfs/images/sub/img02_no_img.gif":
                processed_item["imageUrl"] = image_url

            batch.set(doc_ref, processed_item)

        batch.commit()
        
        success_message = f"{len(final_item_list)}건의 데이터 Firestore에 저장"
        print(success_message)
        return https_fn.Response(success_message, status=200)

    except Exception as e:
        error_message = f"🚨처리 중 오류 발생: {e}"
        print(error_message)
        return https_fn.Response(error_message, status=500)
