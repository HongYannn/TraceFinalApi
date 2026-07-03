DATASET_PATH中的fixed、rand、ches位址請改成本地端存放data資料夾的資料夾位址
MODEL_FOLDER、PIC_OUTPUT_FOLDER也請改成本地端資料夾的位址
在主程式中有三個變數分別是user_model、user_leakage、user_dataset，目前程式是手動修改，由於要跟前端做連結，請改成接收到前端的資料及選擇後轉變成CNN或MLP在丟入user_model變數中，user_leakage同理前端抓到要使用HW或ID後轉變成HW跟ID這兩個關鍵字後丟入變數，而user_dataset同理，關鍵變數名稱為FIXED、RAND、CHES
跑出來的GE分數圖以及pearnsor係數圖皆按照對應的模型名稱存放在pic資料夾中
1. 觸發分析任務 (POST /api/evaluate)
Payload (JSON):
{
  "model_type": "CNN", //CNN OR MLP
  "leakage_model": "HW", //HW OR ID
  "dataset_type": "CHES" //FIXED OR RAND OR CHES
}
JavaScript 請求範例：
async function startEvaluation() {
  const response = await fetch('http://127.0.0.1:8000/api/evaluate', {
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
2. 檢查任務狀態 (GET /api/status)
因為運算會耗費數秒到數十秒，前端每隔 2~3 秒發送一次這個請求，當看到狀態變成 "success" 時，即可把上一步拿到的 ge_url 與 pearson_url 塞進 <img> 標籤中渲染出來！
async function checkStatus() {
  const response = await fetch('http://127.0.0.1:8000/api/status');
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
