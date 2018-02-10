##按照训练数据：验证数据=5:1的比例给数据集做数据划分
import os
from tqdm import tqdm
imgsrc = '$TRAIN_ROOT/train_detected/'
listAll = os.listdir(imgsrc)
#分别计算5类表情的数据数量，按照比例选取与部分做验证集，剩下的做训练集
anList = filter(lambda x: x.startswith('anger'), listAll)
haList = filter(lambda x: x.startswith('happiness'), listAll)
suList = filter(lambda x: x.startswith('surprise'), listAll)
sadList = filter(lambda x: x.startswith('sadness'), listAll)
neList = filter(lambda x: x.startswith('neutral'), listAll)
l = [anList,haList,suList,sadList,neList]
anSize =len(anList)//6
haSize = len(haList)//6
suSize = len(suList)//6
neSize = len(neList)//6
#读取数据信息后制作标签txt文件，后续caffe训练
def makeTxt(ln,idx):
    length = len(ln)//6
    for animg in ln[length:]:
        with open('./train.txt','a') as f:
            f.write(animg+' '+str(idx)+'\n')
    for animg in ln[:length]:
        with open('./val.txt','a') as f:
            f.write(animg+' '+str(idx)+'\n')

for idx,listName in tqdm(enumerate(l)):
    makeTxt(listName,idx)