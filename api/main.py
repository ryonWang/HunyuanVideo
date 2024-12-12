from flask import Flask, request, jsonify, send_file
import os
from pathlib import Path
from datetime import datetime
import time
from loguru import logger

from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.utils.file_utils import save_videos_grid
from api.services.redis_service import RedisService

app = Flask(__name__)

# 全局变量存储模型实例
hunyuan_video_sampler = None

# 初始化Redis服务
redis_service = RedisService()

def init_model():
    """初始化模型"""
    global hunyuan_video_sampler
    
    args = parse_args()
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    return hunyuan_video_sampler.args

@app.route('/api/v1/generate', methods=['POST'])
def generate_video():
    try:
        # 1. 检查请求数据
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'JSON数据格式错误'
            }), 400
            
        # 2. 检查用户token
        user_token = data.get('token')
        if not user_token:
            return jsonify({
                'status': 'error',
                'message': '缺少用户token'
            }), 400
            
        # 3. 检查用户状态
        user_status = redis_service.check_user_status(user_token)
        if user_status and user_status.get('status') == 'processing':
            return jsonify({
                'status': 'error',
                'message': '您有正在处理的任务，请稍后再试'
            }), 429  # Too Many Requests

        # 4. 设置用户状态
        task_id = f"task_{int(time.time())}"
        redis_service.set_user_status(user_token, 'processing', task_id)

        try:
            # 5. 必需参数检查
            if 'prompt' not in data:
                return jsonify({
                    'status': 'error',
                    'message': '请填写数据提示词'
                }), 400
            
            # 6. 参数获取和验证
            prompt = data['prompt']
            if not isinstance(prompt, str) or not prompt.strip():
                return jsonify({
                    'status': 'error',
                    'message': '无效提示词'
                }), 400

            # 7. 可选参数获取和验证
            try:
                height = int(data.get('height', 720))
                width = int(data.get('width', 1280))
                video_length = int(data.get('video_length', 129))
                num_inference_steps = int(data.get('num_inference_steps', 50))
                guidance_scale = float(data.get('guidance_scale', 6.0))
                flow_shift = float(data.get('flow_shift', 7.0))
            except (ValueError, TypeError) as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Parameter type error: {str(e)}'
                }), 400

            # 其他可选参数
            seed = data.get('seed')  # 可以为None
            negative_prompt = data.get('negative_prompt')  # 可以为None
            embedded_guidance_scale = data.get('embedded_guidance_scale')  # 可以为None

            # 8. 参数范围验证
            if height <= 0 or width <= 0:
                return jsonify({
                    'status': 'error',
                    'message': 'Height and width must be positive'
                }), 400

            if video_length <= 0:
                return jsonify({
                    'status': 'error',
                    'message': 'Video length must be positive'
                }), 400

            # 9. 生成视频
            outputs = hunyuan_video_sampler.predict(
                prompt=prompt,
                height=height,
                width=width,
                video_length=video_length,
                seed=seed,
                negative_prompt=negative_prompt,
                infer_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                flow_shift=flow_shift,
                batch_size=1,
                embedded_guidance_scale=embedded_guidance_scale
            )
            
            # 10. 保存视频
            sample = outputs['samples'][0].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_dir = "./results"
            os.makedirs(save_dir, exist_ok=True)
            
            video_name = f"{time_flag}_seed{outputs['seeds'][0]}_{outputs['prompts'][0][:100].replace('/','')}.mp4"
            save_path = os.path.join(save_dir, video_name)
            save_videos_grid(sample, save_path, fps=24)
            
            # 11. 清除用户状态
            redis_service.clear_user_status(user_token)
            
            # 12. 返回结果
            return jsonify({
                'status': 'success',
                'video_path': save_path,
                'seed': outputs['seeds'][0],
                'prompt': outputs['prompts'][0]
            })

        except Exception as e:
            # 发生错误时清除用户状态
            redis_service.clear_user_status(user_token)
            raise e

    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/v1/video/<path:video_name>', methods=['GET'])
def get_video(video_name):
    """获取生成的视频文件"""
    try:
        video_path = os.path.join("./results", video_name)
        if not os.path.exists(video_path):
            return jsonify({
                'status': 'error',
                'message': 'Video file not found'
            }), 404
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error serving video: {str(e)}"
        }), 500

# 添加健康检查接口
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 检查Redis连接
        redis_status = redis_service.client.ping()
        # 检查模型是否已加载
        model_status = hunyuan_video_sampler is not None
        
        return jsonify({
            'status': 'success',
            'redis_connected': redis_status,
            'model_loaded': model_status,
            'server_time': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# 添加测试接口
@app.route('/api/v1/test', methods=['POST'])
def test_generate():
    """测试接口 - 模拟视频生成"""
    try:
        # 1. 检查请求数据
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'JSON数据格式错误'
            }), 400
            
        # 2. 检查用户token
        user_token = data.get('token')
        if not user_token:
            return jsonify({
                'status': 'error',
                'message': '缺少用户token'
            }), 400
            
        # 3. 检查用户状态
        user_status = redis_service.check_user_status(user_token)
        if user_status and user_status.get('status') == 'processing':
            return jsonify({
                'status': 'error',
                'message': '您有正在处理的任务，请稍后再试'
            }), 429
        
        # 4. 模拟处理
        time.sleep(2)  # 模拟处理时间
        
        return jsonify({
            'status': 'success',
            'test_data': data,
            'message': 'Test request successful',
            'server_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Test generate error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # 初始化模型
    args = init_model()
    
    # 启动服务器
    app.run(
        host=os.getenv("SERVER_NAME", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "5000")),
        debug=False  # 生产环境设置为False
    ) 