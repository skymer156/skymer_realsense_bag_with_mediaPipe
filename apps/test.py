def is_inshape(width: int, height: int, pixels) -> bool:
    bool_inshape = True
    if not 0 <= pixels[0] < width:
        bool_inshape = False
    if not 0 <= pixels[1] < height:
        bool_inshape = False
    return bool_inshape


bl = is_inshape(480, 640, [480, 639])
print(bl)
