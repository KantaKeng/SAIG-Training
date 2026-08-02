"""
Backend สำหรับหน้าเว็บ traffic_app.html
รัน: pip install flask flask-cors joblib pandas numpy xgboost --break-system-packages
     python backend_app.py

ก่อนรัน ต้องมีไฟล์ speed_model.json และ speed_model_meta.joblib (จาก save_model() ในโน้ตบุ๊ค) อยู่โฟลเดอร์เดียวกัน
และตั้งค่า LONGDO_API_KEY ให้เป็น key จริงที่มีสิทธิ์ Route Service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # อนุญาตให้หน้าเว็บ (คนละ origin) เรียก API นี้ได้

LONGDO_API_KEY = '561d8b4b88cfdb6b00410d14df56304a'
TRAFFIC_URL = 'https://api.longdo.com/RouteService/json/traffic/speed'
import os

print(os.getcwd())
# โหลดโมเดลที่เทรน+เซฟไว้จากโน้ตบุ๊ค (เซลล์ 6.1: save_model())
# ใช้ native format ของ XGBoost (.json) แทน pickle ทั้งก้อน กันปัญหาเวอร์ชัน xgboost ไม่ตรงกันระหว่างเครื่อง
from xgboost import XGBRegressor
speed_model = XGBRegressor()
speed_model.load_model('speed_model.json')
meta = joblib.load('speed_model_meta.joblib')
FEATURE_COLUMNS = meta['feature_columns']

# บัฟเฟอร์ความเร็วย้อนหลังต่อถนน (อยู่ในหน่วยความจำ รีสตาร์ท backend แล้วจะรีเซ็ต)
speed_history = {}


def fetch_traffic(lat, lon):
    """เรียก Longdo Traffic Speed API (ทำ Coordinate Snapping ให้อัตโนมัติ)"""
    params = {'lat': lat, 'lon': lon, 'range': 0.001, 'locale': 'th', 'key': LONGDO_API_KEY}
    resp = requests.get(TRAFFIC_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_features(road, current_speed, snap_distance, is_raining=0):
    """สร้าง feature ให้ตรงกับตอนเทรนโมเดลทุกประการ"""
    now = datetime.now()
    hist = speed_history.setdefault(road, [])
    hist.append(current_speed)
    speed_history[road] = hist[-4:]

    def lag(n):
        return hist[-1 - n] if len(hist) > n else current_speed

    hour, day = now.hour, now.weekday()

    feat = {
        'Hour': hour,
        'Minute': now.minute,
        'DayOfWeek': day,
        'Is_Rush_Hour': int((7 <= hour <= 9) or (17 <= hour <= 19)),
        'Snapping_Distance': snap_distance,
        'Is_Raining': is_raining,
        'Historical_Speed_1': lag(1),
        'Historical_Speed_2': lag(2),
        'Historical_Speed_3': lag(3),
        'Speed_Rolling_Mean_3': sum(hist[-3:]) / len(hist[-3:]),
        'Current_Speed': current_speed,
        'Hour_Sin': np.sin(2 * np.pi * hour / 24),
        'Hour_Cos': np.cos(2 * np.pi * hour / 24),
        'Day_Sin': np.sin(2 * np.pi * day / 7),
        'Day_Cos': np.cos(2 * np.pi * day / 7),
        'Is_Weekend': int(day >= 5),
    }

    feat_df = pd.DataFrame([feat])
    feat_df[f'RoadID_{road}'] = 1
    feat_df = feat_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return feat_df


def categorize_risk(speed):
    if speed < 20:
        return 'high'
    elif speed <= 45:
        return 'medium'
    return 'low'


@app.route('/analyze', methods=['POST'])
def analyze():
    body = request.get_json()
    lat, lon = body.get('lat'), body.get('lon')

    if lat is None or lon is None:
        return jsonify({'error': 'ต้องส่ง lat และ lon มาด้วย'}), 400

    try:
        data = fetch_traffic(lat, lon)
    except requests.RequestException as e:
        return jsonify({'error': f'เรียก Longdo ไม่สำเร็จ: {e}'}), 502

    if 'meta' in data and data.get('meta', {}).get('error'):
        return jsonify({'error': f"Longdo ไม่มีข้อมูลถนนที่จุดนี้: {data['meta']}"}), 404

    road = data.get('road', 'ไม่ทราบชื่อถนน')
    direction = data.get('dir')
    current_speed = (data.get('speed') or 0) * 3.6  # m/s -> km/h
    matched_lat, matched_lon = data.get('lat'), data.get('lon')

    snap_distance = 0
    if matched_lat is not None and matched_lon is not None:
        R = 6371000
        p1, p2 = np.radians(lat), np.radians(matched_lat)
        dphi = np.radians(matched_lat - lat)
        dl = np.radians(matched_lon - lon)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        snap_distance = R * 2 * np.arcsin(np.sqrt(a))

    feat_df = build_features(road, current_speed, snap_distance)
    predicted_speed = float(speed_model.predict(feat_df)[0])

    return jsonify({
        'road': road,
        'direction': direction,
        'current_speed': current_speed,
        'predicted_speed': predicted_speed,
        'risk_level': categorize_risk(predicted_speed),
        'time': datetime.now().strftime('%H:%M'),
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
