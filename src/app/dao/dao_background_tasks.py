from app.dao.dao_base import DAOBase
from app.config import MONGO_CLIENT, MONGODB_NEW_DB_NAME
from app.models.background_tasks import BackgroundTask, BackgroundTaskInDB



class DAOBackgroundTasks(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_NEW_DB_NAME,
                         'background_tasks',
                         BackgroundTask,
                         BackgroundTaskInDB)

    def insert_one(self, background_task: BackgroundTask):
        return super().insert_one(background_task.dict())

    def replace_one(self, background_task: BackgroundTask):
        return super().replace_one("task_id", background_task.task_id, background_task.dict())

