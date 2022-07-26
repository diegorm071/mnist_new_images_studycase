import pydantic


class WandbAcess(pydantic.BaseSettings):
    WANDB_URL: str

    class Config:
        env_file = ".env"
