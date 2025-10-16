from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
# এই চার লাইনে আমরা প্রয়োজনীয় টুলগুলো নিয়ে আসছি (import করছি)।
# কেন: কারণ আমাদের program-এ এগুলো দরকার—ডেটা রাখার জন্য `pandas`, ওয়েব সার্ভার চালানোর জন্য `Flask`, মেশিন-লার্নিং মডেল জন্য `DecisionTreeClassifier`, আর মডেল সংরক্ষণ/লোড করার জন্য `pickle`।
app = Flask(__name__)
# এখানে আমরা একটা Flask অ্যাপ তৈরি করছি আর সেটা `app` নামেই রাখছি।
# Flask অ্যাপ হল আমাদের ওয়েবসার্ভারের “মস্তিষ্ক” — এটি বলতে পারে “যখন কেউ homepage দেখতে চায়, তখন কী দেখাবো” ইত্যাদি।

dataSet = {
    'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
    'Age': [18, 21, 23, 26, 29, 32, 35, 38, 42, 45, 50, 60],
    'Rent': [6000, 7000, 9000, 10000, 12000, 15000, 17000, 18000, 20000, 22000, 25000, 30000],
    'City': ['Sylhet', 'Dhaka', 'Gazipur', 'Cumilla', 'Tangail', 'Dhaka', 'Barisal', 'Khulna', 'Rangpur', 'Narayanganj', 'Chattogram', 'Dhaka']
}
# এখানে আমরা একটা dictionary বানিয়েছি — টেবিলের মতো ডেটা যেখানে নাম, বয়স, ভাড়া, শহর আছে।
# মডেলকে শেখাতে আমাদের কিছু উদাহরণ দরকার — যেগুলো থেকে সে শিখবে। এই `dataSet` হলো সেই উদাহরণগুলো।

ds = pd.DataFrame(dataSet)
# আমরা ওই dictionary-কে `pandas`-এর `DataFrame` এ পরিণত করছি — অর্থাৎ টেবিল আকারে।
# DataFrame খুবই সুবিধাজনক — কলাম ও রো দেখে সহজে কাজ করা যায় (filter, group, apply ইত্যাদি)।
def suggest_room(age):
    if age < 22:
        return 'Shared Room'
    elif 22 <= age <= 28:
        return 'Single Room'
    elif 29 <= age <= 40:
        return 'Double Room'
    else:
        return 'Family Room'
# এটা একটা function (ছোট ছোট নিয়ম) — `age` দিলে বলবে কোন রুম টাইপ উপযুক্ত।
# আমরা training data-র জন্য label (target) তৈরি করতে চাই — অর্থাৎ প্রতিটি বয়সের জন্য কোন রুম হবে সেটা নির্ধারণ করতে। এই function সেই নিয়ম দেয়।

ds['RoomType'] = ds['Age'].apply(suggest_room)
# DataFrame-এর প্রতিটি `Age`-এ `suggest_room()` function চালিয়ে ফলাফল `RoomType` নামে নতুন কলামে বসাচ্ছে।
# এইভাবে আমাদের dataset-এ প্রপার label (RoomType) যোগ হবে, যাতে মডেল তা দেখে শেখতে পারে।
X = ds[['Age']]
y = ds['RoomType']
# এখানে আমরা মডেলকে কি দেখাবো (input) আর কি শেখাবো (output) সেটি আলাদা করছি:
# * `X` = input features = `Age` (একটা টেবিল/কলাম হিসেবে)
# * `y` = target label = `RoomType`
# Machine Learning লাইব্রেরি আশা করে input আর output আলাদা করে দেওয়া হবে।

model = DecisionTreeClassifier()
model.fit(X, y)
# * প্রথম লাইনে আমরা Decision Tree নামে একটি মডেল তৈরি করলাম।
# * দ্বিতীয় লাইনে `model.fit(X, y)` দিয়ে আমরা মডেলটাকে শেখাচ্ছি — অর্থাৎ `Age` দেখে `RoomType` কী হবে তা শিখতে বলছি।
# `fit` হল training step — মডেল ডেটা পড়ে এবং সিদ্ধান্ত নেওয়ার নিয়ম internally তৈরি করে (একটা ট্রি গঠন করে)।

# # চাইলে pickle করে সংরক্ষণ করতে পারো
pickle.dump(model, open('model.pkl', 'wb'))
# এখানে আমরা `model`-টাকে `model.pkl` নামে ফাইলে সেভ করে রাখছি। `wb` মানে write-binary।
# একবার মডেল train করে রাখলে বারবার train করার দরকার নেই — পরে ওই সেভ করা ফাইল লোড করে prediction করা যাবে। এটা সময় বাঁচায়।

# # ======================
# # ROUTE: Homepage
# ======================
@app.route('/')
def home():
    return render_template('index.html')
# এই অংশটি বলে — “যখন কেউ ব্রাউজারে ওয়েবসাইটের মূল ঠিকানায় (/) যাবে, তখন `index.html` ফাইল দেখাবে”।
# আমাদের UI (form) একটা HTML পেজে আছে; ওই পেজকে browser-এ দেখানোর জন্য route লাগবে।

# # ======================
# # ROUTE: Prediction
# ======================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        prediction = model.predict([[age]])[0]
        return render_template('index.html', prediction_text=f"Suggested Room: {prediction}")
    except:
        return render_template('index.html', prediction_text="Please enter a valid age.")
# এখানে তিনটি অংশে ভাগ করছি — line-by-line আলাদা ব্যাখ্যা:
# 1. `@app.route('/predict', methods=['POST'])`
#     এটি বলে — “/predict” ঠিকানায় যদি কেউ data পাঠায় (form submit করে), সেই request এখানে আসবে। `methods=['POST']` মানে আমরা ডেটা লুকিয়ে পাঠাই (form-এর মাধ্যমে)।
# 2. `age = int(request.form['age'])`
#    browser থেকে যেই form data এসেছে সেটা আমরা নিচ্ছি: `request.form['age']` — এটা string হিসেবে আসে; `int()` করে number এ রূপান্তর করছি। model-এ prediction দিতে integer (সংখ্যা) দরকার।
# 3. `prediction = model.predict([[age]])[0]`
#    মডেলকে prediction করার জন্য `model.predict()` কল করছি এবং আউটপুট নিচ্ছি। `predict()` expects 2D array — তাই `[[age]]` ব্যবহার করি (একটা row, একট কলাম)। `[0]` দিয়ে prediction string (যেমন "Single Room") বের করছি।
# 4. `return render_template('index.html', prediction_text=f"Suggested Room: {prediction}")`
#    prediction পাইলে আবার `index.html` রেন্ডার করে পাঠাচ্ছি এবং `prediction_text` নামের একটা ভ্যারিয়েবল পাঠাচ্ছি যাতে HTML পেজে তা দেখানো যায়।
# 5. `except:` … error handling
#    যদি কোনো ভুল হয় (যেমন ইউজার age না দিয়েছে বা লেখার সময় ভুল), তখন আমরা একই পেজে একটা friendly message দেখাই — "Please enter a valid age." যাতে program crash না করে এবং ইউজার জানে কি ভুল হয়েছে।

if __name__ == "__main__":
    app.run(debug=True)
# * `if __name__ == "__main__":` মানে এই ফাইল সরাসরি চালানো হলে নিচের কোড চালবে (আর অন্য কোথাও import করলে চলবে না)।
# * `app.run(debug=True)` Flask development server চালায়। `debug=True` দিলে two things হয়:
#   1. কোড পরিবর্তন করলে সার্ভার automatic রিলোড করে।
#   2. কোনো error হলে browser-এ সুন্দর debug page দেখায় (চূড়ান্ত ডিপ্লয়মেন্টে এটা বন্ধ করা উচিত)।

# ## একসাথে পুরো কাজটা কি করে?
# 1. তুমি Python চালাও → Flask server চালু হয়।
# 2. ব্রাউজারে `http://127.0.0.1:5000/` গেলে `index.html` আসে, যেখানে age দেওয়ার form আছে।
# 3. ইউজার age দেয়, submit করে — browser `/predict` এ POST request পাঠায়।
# 4. Flask `predict()` function age নেয়, মডেলকে বলে “এটা predict কর”।
# 5. মডেল পূর্বে শেখা অনুযায়ী রুম টাইপ বলে দেয় (e.g., "Single Room")।
# 6. Flask সেই ফলাফল নিয়ে `index.html` আবার রেন্ডার করে এবং ফলাফল দেখায়।