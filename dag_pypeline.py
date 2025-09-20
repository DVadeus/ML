from dagster import job, op, Definitions

@op
def say_hello():
    return "Hola desde Dagster en Render"

@job
def hello_job():
    say_hello()

defs = Definitions(jobs=[hello_job])

