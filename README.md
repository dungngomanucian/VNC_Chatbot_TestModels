# Cài biến môi trường
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Lấy HF_Token từ access tokens trong tài khoản hugging face và lưu vào .env
# read_test_file là Script helper để đọc các file test .txt và trả về query_chunk và chunks