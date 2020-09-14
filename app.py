from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('pred.pkl', 'rb'))
an = pickle.load(open('an.pkl', 'rb'))
ap = pickle.load(open('ap.pkl', 'rb'))
ar = pickle.load(open('ar.pkl', 'rb'))
assam = pickle.load(open('as.pkl', 'rb'))
br = pickle.load(open('br.pkl', 'rb'))
ch = pickle.load(open('ch.pkl', 'rb'))
ct = pickle.load(open('ct.pkl', 'rb'))
dl = pickle.load(open('dl.pkl', 'rb'))
dndd = pickle.load(open('dndd.pkl', 'rb'))
ga = pickle.load(open('ga.pkl', 'rb'))
gj = pickle.load(open('gj.pkl', 'rb'))
hp = pickle.load(open('hp.pkl', 'rb'))
hr = pickle.load(open('hr.pkl', 'rb'))
jh = pickle.load(open('jh.pkl', 'rb'))
jk = pickle.load(open('jk.pkl', 'rb'))
ka = pickle.load(open('ka.pkl', 'rb'))
kl = pickle.load(open('kl.pkl', 'rb'))
la = pickle.load(open('la.pkl', 'rb'))
mh = pickle.load(open('mh.pkl', 'rb'))
ml = pickle.load(open('ml.pkl', 'rb'))
mn = pickle.load(open('mn.pkl', 'rb'))
mp = pickle.load(open('mp.pkl', 'rb'))
mz = pickle.load(open('mz.pkl', 'rb'))
nl = pickle.load(open('nl.pkl', 'rb'))
orissa = pickle.load(open('or.pkl', 'rb'))
pb = pickle.load(open('pb.pkl', 'rb'))
py = pickle.load(open('py.pkl', 'rb'))
rj = pickle.load(open('rj.pkl', 'rb'))
sk = pickle.load(open('sk.pkl', 'rb'))
tg = pickle.load(open('tg.pkl', 'rb'))
tn = pickle.load(open('tn.pkl', 'rb'))
tr = pickle.load(open('tr.pkl', 'rb'))
up = pickle.load(open('up.pkl', 'rb'))
ut = pickle.load(open('ut.pkl', 'rb'))
wb = pickle.load(open('wb.pkl', 'rb'))

transf = pickle.load(open('transform.pkl', 'rb'))


app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    place = request.form['b']
    
    data1 = request.form['a']
    
    month = request.form['c']
    if month.lower() == 'september':
        data2 = int(data1) + int(214)
    else:
        data2 = int(data1) + int(244)
    
    transformed_data = transf.transform([[data2]])
    
    if place.lower() == 'india':
        pred = model.predict(transformed_data)
    elif (place.lower() == 'andman and nicobar') | (place.lower() == 'andman & nicobar'):
        pred = an.predict(transformed_data)
    elif place.lower() == 'andhra pradesh':
        pred = ap.predict(transformed_data)
    elif place.lower() == 'arunachal pradesh':
        pred = ar.predict(transformed_data)
    elif place.lower() == 'assam':
        pred = assam.predict(transformed_data)
    elif place.lower() == 'bihar':
        pred = br.predict(transformed_data)
    elif place.lower() == 'chandigarh':
        pred = ch.predict(transformed_data)
    elif place.lower() == 'chhattisgarh':
        pred = ct.predict(transformed_data)
    elif (place.lower() == 'dadar and nagar haveli') | (place.lower == 'daman and diu'):
        pred = dndd.predict(transformed_data)
    elif place.lower() == 'delhi':
        pred = dl.predict(transformed_data)
    elif place.lower() == 'goa':
        pred = ga.predict(transformed_data)
    elif place.lower() == 'gujrat':
        pred = gj.predict(transformed_data)
    elif place.lower() == 'haryana':
        pred = hr.predict(transformed_data)
    elif place.lower() == 'himachal pradesh':
        pred = hp.predict(transformed_data)
    elif (place.lower() == 'jammu and kashmir') | (place.lower() == 'jammu & kashmir'):
        pred = jk.predict(transformed_data)
    elif place.lower() == 'jharkhand':
        pred = jh.predict(transformed_data)
    elif place.lower() == 'karnataka':
        pred = ka.predict(transformed_data)
    elif place.lower() == 'kerala':
        pred = kl.predict(transformed_data)
    elif place.lower() == 'madhya pradesh':
        pred = mp.predict(transformed_data)
    elif place.lower() == 'maharashtra':
        pred = mh.predict(transformed_data)
    elif place.lower() == 'manipur':
        pred = mn.predict(transformed_data)
    elif place.lower() == 'meghalaya':
        pred = ml.predict(transformed_data)
    elif place.lower() == 'mizoram':
        pred = mz.predict(transformed_data)
    elif place.lower() == 'nagaland':
        pred = nl.predict(transformed_data)
    elif (place.lower() == 'odisha') | (place.lower() == 'orissa'):
        pred = orissa.predict(transformed_data)
    elif place.lower() == 'puducherry':
        pred = py.predict(transformed_data)
    elif place.lower() == 'punjab':
        pred = pb.predict(transformed_data)
    elif place.lower() == 'rajasthan':
        pred = rj.predict(transformed_data)
    elif place.lower() == 'sikkim':
        pred = sk.predict(transformed_data)
    elif place.lower() == 'tamil nadu':
        pred = tn.predict(transformed_data)
    elif place.lower() == 'telangana':
        pred = tg.predict(transformed_data)
    elif place.lower() == 'tripura':
        pred = tr.predict(transformed_data)
    elif place.lower() == 'uttar pradesh':
        pred = up.predict(transformed_data)
    elif place.lower() == 'uttarakhand':
        pred = ut.predict(transformed_data)
    elif place.lower() == 'west bengal':
        pred = wb.predict(transformed_data)
    else:
        pred = 'wrong input'
    
    return render_template('after.html', data=int(pred[0,0]), date = data1, month = month, place = place.lower())


if __name__ == "__main__":
    app.run(debug=False)















