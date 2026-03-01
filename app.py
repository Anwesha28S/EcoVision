from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model('waste_classification_model.h5')

classes = ['Hazardous', 'Non-Recyclable', 'Organic', 'Recyclable']
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
RECYCLING_INFO = {
    'Hazardous': {
        'advice': 'Hazardous waste (batteries, chemicals) requires special disposal. Do not throw in regular bins!',
        'query': 'how to dispose of hazardous household waste safely'
    },
    'Non-Recyclable': {
        'advice': 'This item cannot be recycled easily. Try to reduce use or dispose of in general waste.',
        'query': 'what to do with non recyclable waste'
    },
    'Organic': {
        'advice': 'Organic waste is great for composting! Use it to make nutrient-rich soil for plants.',
        'query': 'how to start composting organic waste at home'
    },
    'Recyclable': {
        'advice': 'Clean and dry this item before placing it in the recycling bin.',
        'query': 'how to properly recycle paper plastic and glass'
    }
}











    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        save_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(save_path)

        #Preprocessing
        img = image.load_img(save_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array)
        result = classes[np.argmax(preds)]
        
        info = RECYCLING_INFO.get(result, {"advice": "Follow local guidelines.", "query": "recycling tips"})
        yt_search_url = f"https://www.youtube.com/results?search_query={info['query'].replace(' ', '+')}"
        return render_template('index.html', prediction=result, img_path=save_path,advice=info['advice'],
                               yt_url=yt_search_url,)
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'): os.makedirs('uploads')
    app.run(debug=True)