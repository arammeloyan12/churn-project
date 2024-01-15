from pydantic_settings import BaseSettings, SettingsConfigDict


class SQLConfig(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASS: str
    DB_NAME: str 

    
    @property
    def database_url_psycopg(self):
        # pastgresql+psycopg://postgres:postgres@localhost:5432/db_name
        return f"""postgresql+psycopg2://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:
                {self.DB_PORT}/{self.DB_NAME}"""

    model_config = SettingsConfigDict(env_file=".env")
