##����ѵ�����ݣ���֤����=5:1�ı��������ݼ������ݻ���
import os
from tqdm import tqdm
imgsrc = '$TRAIN_ROOT/train_detected/'
listAll = os.listdir(imgsrc)
#�ֱ����5�������������������ձ���ѡȡ�벿������֤����ʣ�µ���ѵ����
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
#��ȡ������Ϣ��������ǩtxt�ļ�������caffeѵ��
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