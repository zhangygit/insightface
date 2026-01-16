import cv2
import numpy as np
import gc
from fastapi import FastAPI, File, UploadFile, HTTPException
from insightface.app import FaceAnalysis
from loguru import logger
import uvicorn
import os
app = FastAPI(title="高性能牛脸识别服务")

logger.add("log/app.log", rotation="100 MB", retention="7 days", level="INFO")

MODEL_ROOT = os.getenv("MODEL_ROOT",  "./models")

model = FaceAnalysis(root=MODEL_ROOT, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))


@app.post("/detection")
async def detect_faces(file: UploadFile = File(...)):
  img = None # 预设变量
  logger.info(f"收到检测请求: {file.filename}")
  if not file.content_type.startswith("image/"):
    logger.warning(f"Invalid file type: {file.content_type}")
    raise HTTPException(status_code=400, detail="File must be an image.")
  try:
    # 1. 读取上传的图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
      raise ValueError("图片解码失败")

    # 2. 推理
    num = model.detect(img)
    logger.info(f"检测完成 | 牛脸数: {num} ")
    return {"num": num}
  except Exception as e:
    logger.exception(f"检测接口内部错误: {e}")
    raise HTTPException(status_code=500, detail=str(e))
  finally:
    # --- 强制内存清理 ---
    if img is not None:
      del img
    gc.collect()

@app.post("/recognition")
async def recognition(file: UploadFile = File(...)):
  img = None
  logger.info(f"收到识别请求: {file.filename}")
  if not file.content_type.startswith("image/"):
    logger.warning(f"Invalid file type: {file.content_type}")
    raise HTTPException(status_code=400, detail="File must be an image.")
  try:
    # 1. 读取上传的图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
      raise ValueError("图片解码失败")

    # 2. 推理
    faces = model.get(img)
    embs= [face.normed_embedding.tolist() for face in faces]

    logger.info(f"识别完成 | 提取牛数: {len(embs)}")

    return {"embedding": embs}
  except Exception as e:
    logger.exception(f"识别接口内部错误: {e}")
    raise HTTPException(status_code=500, detail=str(e))
  finally:
    # --- 强制内存清理 ---
    if img is not None:
      del img
    gc.collect()


if __name__ == "__main__":
  logger.info("服务启动成功")
  uvicorn.run(app, host="0.0.0.0", port=5004)