//server
uvicorn gateway:app --host 0.0.0.0 --port 4000

//AI
uvicorn service:app --host 0.0.0.0 --port 5001
uvicorn app.main:app --host 0.0.0.0 --port 5002
    //AI t23d -- ran simple_image2mesh_server
    python simple_image2mesh_server.py


//NGROK
ngrok http 4000


//NGROK
ngrok http 4000

//server
uvicorn gateway:app --host 0.0.0.0 --port 4000

//AI
cd/nsfw-image-detect
uvicorn service:app --host 0.0.0.0 --port 5001

cd/blip-image-captioning-api-main
uvicorn app.main:app --host 0.0.0.0 --port 5002

cd/3dgen/Hunyuan3D-2-main
python simple_image2mesh_server.py


