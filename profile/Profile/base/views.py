#필요한 라이브러리
import os
import numpy as np
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse


from keras.models import load_model


# 유저가 요청하면 홈페이지로 연결한다
# Create your views here.
def home(request):
    return render(request, 'home.html')

def projects(request):
    return render(request, 'projects.html')

def contact(request):
    return render(request, 'contact.html')

def mnist(request):
    file = request.FILES.get("file")
    if not file:
        # return HttpResponse("업로드 되지 않았습니다.")
        return render(request, "mnist.html")
    else:
        file_name = file.name
        file_path = os.path.join(settings.MEDIA_ROOT, 'images', file_name)

        with default_storage.open(file_path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        #테스트 이미지 준비
        test_image = Image.open(file_path)
        test_image = test_image.resize(28,28)
        test_image = test_image.convert('L') 
        test_image = np.asarray(test_image)
        test_image = test_image.reshape(1, 28, 28, 1)
        
        
        #모델을 이용해서 예측
        model = load_model('C:/Users/tyufg/바탕화면/CNN/Profile/mnist_model.h5',compile=False)
        digit = np.argmax(model.predict(test_image), axis=1)[0]
        probs = model.predict(test_image)[0][digit]

        return render(request, "mnist.html", {"digit": digit ,"probs": probs})
        # return HttpResponse("업로드 되었습니다.")


def fashion_mnist(request):
    file = request.FILES.get("file")
    if not file:
        return render(request, "fashion_mnist.html")
    else:
        file_name = file.name
        file_path = os.path.join(settings.MEDIA_ROOT, 'images',
                                  file_name)

        with default_storage.open(file_path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        #테스트 이미지 준비
        test_image = Image.open(file_path)
        test_image = test_image.resize(28,28)
        test_image = test_image.convert('L') 
        test_image = np.asarray(test_image)
        test_image = 255- test_image
        test_image = test_image / 255.0
        test_image = test_image.reshape(1, 28, 28, 1)
        
        
        # 예측
        model = load_model("base/models/fashion_mnist_model.h5",compile=False)
        clothing_num = np.argmax(model.predict(test_image), axis=1)[0]
        
        # 라벨이름 저장
        label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        clothing = label_names[clothing_num]
        probs = model.predict(test_image)[0][clothing_num]
    

    return render(request, "fashion_mnist.html",{"clothing": clothing ,
                                                 "probs": probs})
def visualization(request):
    return render(request, "visualization.html")
