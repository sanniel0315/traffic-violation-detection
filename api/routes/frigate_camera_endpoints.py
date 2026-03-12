"""
新增以下兩個端點到 api/routes/frigate.py
貼在現有的路由之後即可
"""

# ========== 新增攝影機到 Frigate ==========

class FrigateCameraAdd(BaseModel):
    """新增 Frigate 攝影機"""
    name: str
    rtsp_url: str
    record: bool = True
    detect: bool = True
    snapshots: bool = True


@router.post("/camera")
async def add_frigate_camera(camera: FrigateCameraAdd):
    """新增攝影機到 Frigate config.yml"""
    try:
        config_path = Path(FRIGATE_CONFIG_PATH)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Frigate 設定檔不存在")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        if 'cameras' not in config:
            config['cameras'] = {}

        # 檢查名稱是否已存在
        if camera.name in config['cameras']:
            return {"status": "error", "message": f"攝影機 {camera.name} 已存在"}

        # 建立攝影機設定
        cam_config = {
            'enabled': True,
            'ffmpeg': {
                'inputs': [{
                    'path': camera.rtsp_url,
                    'roles': ['detect']
                }]
            },
            'detect': {
                'enabled': camera.detect,
                'width': 1920,
                'height': 1080,
                'fps': 10
            },
            'objects': {
                'track': ['car', 'motorcycle', 'bicycle', 'person']
            }
        }

        # 如果開啟錄影，加入 roles 和 record 設定
        if camera.record:
            cam_config['ffmpeg']['inputs'][0]['roles'].append('record')
            cam_config['record'] = {
                'enabled': True,
                'retain': {
                    'days': 7,
                    'mode': 'motion'
                }
            }

        # 如果開啟截圖
        if camera.snapshots:
            cam_config['snapshots'] = {
                'enabled': True,
                'bounding_box': True,
                'timestamp': True,
                'retain': {
                    'default': 14
                }
            }

        config['cameras'][camera.name] = cam_config

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return {
            "status": "success",
            "message": f"已新增攝影機 {camera.name}，請重啟 Frigate 使設定生效"
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": f"新增失敗: {str(e)}"}


# ========== 刪除 Frigate 攝影機 ==========

@router.delete("/camera/{name}")
async def delete_frigate_camera(name: str):
    """從 Frigate config.yml 刪除攝影機"""
    try:
        config_path = Path(FRIGATE_CONFIG_PATH)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Frigate 設定檔不存在")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        cameras = config.get('cameras', {})
        if name not in cameras:
            return {"status": "error", "message": f"找不到攝影機 {name}"}

        del cameras[name]
        config['cameras'] = cameras

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return {
            "status": "success",
            "message": f"已刪除攝影機 {name}，請重啟 Frigate 使設定生效"
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": f"刪除失敗: {str(e)}"}
