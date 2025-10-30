const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

app.use(express.static(path.join(__dirname, 'frontend')));

app.get('/api/keyword_query', (req, res) => {
  const keyword = req.query.keyword;
  
  if (!keyword) {
    return res.status(400).json({ error: '缺少关键词参数' });
  }

  console.log(`接收到关键词: ${keyword}`);
  
  const pythonScriptPath = path.join(__dirname, 'backend/keyword_query.py');
  
  const pythonProcess = spawn('python', [pythonScriptPath, keyword]);
  
  let resultData = '';
  let errorData = '';
  
  pythonProcess.stdout.on('data', (data) => {
    resultData += data.toString('utf8');
  });
  
  pythonProcess.stderr.on('data', (data) => {
    errorData += data.toString('utf8');
  });
  
  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Python脚本执行错误: ${errorData}`);
      return res.status(500).json({ error: '脚本执行失败', details: errorData });
    }
    
    const jsonResult = JSON.parse(resultData);
    console.log(`Python返回数据:`, jsonResult);
    
    res.json(jsonResult);
  });
});
app.get('/api/init_data', (req, res) => {
  const pythonScriptPath = path.join(__dirname, 'backend/init_data.py');
  
  const pythonProcess = spawn('python', [pythonScriptPath]);
  
  let resultData = '';
  let errorData = '';
  
  pythonProcess.stdout.on('data', (data) => {
    resultData += data.toString('utf8');
  });
  
  pythonProcess.stderr.on('data', (data) => {
    errorData += data.toString('utf8');
  });
  
  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Python脚本执行错误: ${errorData}`);
      return res.status(500).json({ error: '脚本执行失败', details: errorData });
    }
    
    const jsonResult = JSON.parse(resultData);
    console.log(`Python返回数据:`, jsonResult);
    
    res.json(jsonResult);
  });
});

app.listen(PORT, () => {
  console.log(`服务器运行在 http://localhost:${PORT}`);
  console.log(`前端页面: http://localhost:${PORT}/demo.html`);
});