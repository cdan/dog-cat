FROM python:latest

RUN pip install Flask tensorflow matplotlib Pillow

COPY src/ /app/

WORKDIR /app

CMD [ "python", "./app_display_image.py" ]
