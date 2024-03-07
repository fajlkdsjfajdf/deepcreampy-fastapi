import io

from PIL import Image
from fastapi import FastAPI, File
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, Response
from detector import detector
from decensor import decensor
from esrgan import esrgan


app = FastAPI(title="deepcreampy-API")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


def send_image(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return Response(content=image_bytes.getvalue(), media_type="image/png")


@app.get("/", include_in_schema=False)
async def route_index():
    return RedirectResponse("/docs")


@app.post("/deepcreampy-bar-rcnn", summary="修复一个带有色条的图片(自动识别涂抹)",
          description="使用Mask_RCNN 自动识别色条区域并涂抹")
async def route_bar_rcnn(image: bytes = File()) -> Response:
    image = detector(image, is_mosaic=False)
    image = Image.open(io.BytesIO(image))
    result = decensor(image, image, is_mosaic=False)
    return send_image(result)


@app.post("/deepcreampy-mosaic-rcnn", summary="修复一个带有马赛克的图片(自动识别涂抹)",
          description="使用Mask_RCNN 自动识别马赛克区域并涂抹")
async def route_mosaic_rcnn(image: bytes = File()) -> Response:
    org = Image.open(io.BytesIO(image))
    masked_image = detector(image, is_mosaic=True)
    masked = Image.open(io.BytesIO(masked_image))
    result = decensor(org, masked, is_mosaic=True)
    return send_image(result)


@app.post("/deepcreampy-bar-rcnn-esrgan", summary="修复一个带有马赛克的图片(自动识别涂抹并放大)",
          description="使用Mask_RCNN 自动识别马赛克区域并涂抹, 另外由于马赛克区域仍保留原图的形状信息, 用esrgan超分马赛克区域也许可以获得比较好的效果")
async def route_mosaic_rcnn_esrgan(image: bytes = File()) -> Response:
    org = Image.open(io.BytesIO(image))
    org, masked = esrgan(image)
    org = Image.open(io.BytesIO(org))
    masked = Image.open(io.BytesIO(masked))
    result = decensor(org, masked, is_mosaic=True)
    return send_image(result)


@app.post("/deepcreampy-bar", summary="修复一个带有色条的图片(已手动涂抹)")
async def route_bar(image: bytes = File()) -> Response:
    image = Image.open(io.BytesIO(image))
    result = decensor(image, image, is_mosaic=False)
    return send_image(result)


@app.post("/deepcreampy-mosaic", summary="修复一个带有马赛克的图片(已手动涂抹)")
async def route_mosaic(image: bytes = File(), masked: bytes = File()) -> Response:
    image = Image.open(io.BytesIO(image))
    masked = Image.open(io.BytesIO(masked))
    result = decensor(image, masked, is_mosaic=True)
    return send_image(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

