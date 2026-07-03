# Side-Channel Analysis API 串接說明文件

本專案提供側道攻擊（Side-Channel Analysis）分析與評估的網頁 API 服務，支援 CNN / MLP 模型對應不同資料集與洩漏模型的推論，並將分析結果自動繪圖儲存。

---

## 📂 後端環境與路徑配置
在部署後端服務前，請確保 `api_server.py` 內的以下路徑已正確配置為本機端存放路徑：
* **`DATASET_PATHS`**：包含 `fixed`、`rand`、`ches` 的 `.h5` 資料夾與檔案位址。
* **`MODEL_FOLDER`**：存放已訓練模型權重（`.h5`）的資料夾位址。
* **`PIC_OUTPUT_FOLDER`**：設定為 `./pic` 或本地端存放產出圖片的資料夾位址。

### 🔄 動態變數對接機制
本 API 已將原本手動修改的變數（`user_model`、`user_leakage`、`user_dataset`）改為**動態接收前端請求**：
1. **模型架構**：後端接收前端傳入的參數後，會動態轉換為 `CNN` 或 `MLP` 丟入 `user_model` 變數。
2. **洩漏模型**：前端抓取使用者選擇的 `HW` 或 `ID` 後，後端會自動識別這兩個關鍵字並丟入變數中。
3. **資料集選擇**：後端根據前端傳遞的關鍵變數名稱 `FIXED`、`RAND`、`CHES` 自動加載對應的測試集。

所有的 GE 分數圖以及 Pearson 相關係數點位圖，皆會自動按照對應的模型名稱儲存於本機的 `pic/` 資料夾內，並映射至靜態網址供前端讀取。

---

## 🌐 前端串接 API 端點說明

### 1. 觸發分析任務

透過 `POST` 請求通知後端在背景啟動深度學習分析與繪圖任務。此端點為非同步處理，呼叫後後端會立即回傳預期產出的圖片網址。

* **HTTP 方法**：`POST`
* **API 路徑**：`http://127.0.0.1:8000/api/evaluate`
* **請求格式 (Payload JSON)**：
  ```json
  {
    "model_type": "CNN",
    "leakage_model": "HW",
    "dataset_type": "CHES"
  }
model_type 可選值："CNN" 或 "MLP"

leakage_model 可選值："HW" 或 "ID"

dataset_type 可選值："FIXED"、"RAND"、"CHES"

JavaScript 請求範例：
```
async function startEvaluation() {
  const response = await fetch('[http://127.0.0.1:8000/api/evaluate](http://127.0.0.1:8000/api/evaluate)', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_type: "CNN",
      leakage_model: "HW",
      dataset_type: "CHES"
    })
  });

  const data = await response.json();
  console.log("預期的圖片網址：", data.expected_urls);
  // 前端拿到網址後可以先轉圈圈 (loading)，並定時向下方 status API 輪詢進度
}
```
2. 檢查任務狀態
由於後端模型推論與相關係數運算會耗費數秒到數十秒，前端成功觸發任務後，應每隔 2~3 秒發送一次此 GET 請求進行狀態輪詢（Polling）。

HTTP 方法：GET

API 路徑：http://127.0.0.1:8000/api/status

狀態處置說明：當看到狀態變成 "success" 時，即可把上一步從 expected_urls 拿到的 ge_url 與 pearson_url 塞進 <img> 標籤中渲染出來！

JavaScript 請求範例：
```
async function checkStatus() {
  const response = await fetch('[http://127.0.0.1:8000/api/status](http://127.0.0.1:8000/api/status)');
  const data = await response.json();

  if (data.status === "success") {
    console.log("計算完成，可以安全地載入圖片囉！");
    // 更新 UI 顯示圖片
  } else if (data.status === "failed") {
    console.error("錯誤原因:", data.message);
  } else {
    console.log("目前狀態:", data.message);
  }
}
```
