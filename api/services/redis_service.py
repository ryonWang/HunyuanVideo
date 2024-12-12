from redis import Redis
from datetime import datetime
import os
from loguru import logger

class RedisService:
    def __init__(self):
        self.client = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
        self.USER_STATUS_PREFIX = "user:status:"
        self.TASK_EXPIRE_TIME = 3600  # 1小时过期
        
    def check_user_status(self, user_token: str) -> dict:
        """检查用户状态"""
        try:
            key = f"{self.USER_STATUS_PREFIX}{user_token}"
            status = self.client.hgetall(key)
            return status
        except Exception as e:
            logger.error(f"Redis check_user_status error: {str(e)}")
            return {}
            
    def set_user_status(self, user_token: str, status: str, task_id: str) -> bool:
        """设置用户状态"""
        try:
            key = f"{self.USER_STATUS_PREFIX}{user_token}"
            self.client.hmset(key, {
                "status": status,
                "task_id": task_id,
                "start_time": datetime.now().isoformat()
            })
            self.client.expire(key, self.TASK_EXPIRE_TIME)
            return True
        except Exception as e:
            logger.error(f"Redis set_user_status error: {str(e)}")
            return False
            
    def clear_user_status(self, user_token: str) -> bool:
        """清除用户状态"""
        try:
            key = f"{self.USER_STATUS_PREFIX}{user_token}"
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis clear_user_status error: {str(e)}")
            return False 