import torch
import torch.nn as nn
import numpy as np
import random
import os
import time


def loadfile(filepath,batch_size):
    xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
    x_data=torch.from_numpy(xy[0:batch_size,:-1])
    y_data=torch.from_numpy(xy[0:batch_size,[-1]])
    return x_data,y_data

def DMVTSVM(lr,mu1,mu2,mu3,mu4,epoch_num,round_num,num_num,k,batch_size,scale,file,seed_sep,seed_init,vote_num,times):
    randarray=torch.randperm(5*round_num*epoch_num*num_num+10)#用于每个epoch生层不同随机种子
    class autoencoder1(torch.nn.Module):
        def __init__(self):
            super(autoencoder1,self).__init__()
           #encoder
            self.linear1=nn.Linear(76,64)
            self.linear2=nn.Linear(64,32)
            #decoder
            self.dlinear1=nn.Linear(32,64)
            self.dlinear2=nn.Linear(64,76)
            
            self.norm1=torch.nn.BatchNorm1d(76)
            self.norm2=torch.nn.BatchNorm1d(64)
            self.norm3=torch.nn.BatchNorm1d(32)
    
            self.relu=torch.nn.ReLU()    
        def forward(self,x):
            #encoder
            x=self.norm1(x)
            x=self.relu(self.norm2(self.linear1(x)))
            x=self.norm3(self.linear2(x))
            #decoder
            code=self.relu(self.norm2(self.dlinear1(x)))
            code=self.dlinear2(code)
            return x,code
    model1=autoencoder1().to(device)
    
    class SVM11(torch.nn.Module):
        def __init__(self):
            super(SVM11,self).__init__()
            self.linear1=torch.nn.Linear(32,1)
            
        def forward(self,x):
            x=self.linear1(x)
            return x
    svm11=SVM11().to(device)
    
    class SVM12(torch.nn.Module):
        def __init__(self):
            super(SVM12,self).__init__()
            self.linear1=torch.nn.Linear(32,1)
            
        def forward(self,x):
            x=self.linear1(x)
            return x
    svm12=SVM12().to(device)
        
    class autoencoder2(torch.nn.Module):
        def __init__(self):
            super(autoencoder2,self).__init__()
            #encoder
            self.linear1=torch.nn.Linear(64,60)
            self.linear2=torch.nn.Linear(60,32)
            self.dlinear1=torch.nn.Linear(32,60)
            self.dlinear2=torch.nn.Linear(60,64)
            
            self.norm1=torch.nn.BatchNorm1d(64)
            self.norm2=torch.nn.BatchNorm1d(60)
            self.norm3=torch.nn.BatchNorm1d(32)
            #self.norm4=torch.nn.BatchNorm1d(128)
    
            self.relu=torch.nn.ReLU()    
        def forward(self,x):
            #encoder
            x=self.norm1(x)
            x=self.relu(self.norm2(self.linear1(x)))
            x=self.norm3(self.linear2(x))
            code=self.norm2(self.dlinear1(x))
            code=self.dlinear2(code)
            return x,code
    model2=autoencoder2().to(device)
    
    class SVM21(torch.nn.Module):
        def __init__(self):
            super(SVM21,self).__init__()
            self.linear1=torch.nn.Linear(32,1)
            
        def forward(self,x):
            x=self.linear1(x)
            return x
    svm21=SVM21().to(device)
    
    class SVM22(torch.nn.Module):
        def __init__(self):
            super(SVM22,self).__init__()
            self.linear1=torch.nn.Linear(32,1)
            
        def forward(self,x):
            x=self.linear1(x)
            return x
    svm22=SVM22().to(device)
    
    class autoencoder3(torch.nn.Module):
        def __init__(self):
            super(autoencoder3,self).__init__()
            #encoder
            self.linear1=torch.nn.Linear(216,200)
            self.linear2=torch.nn.Linear(200,128)
            self.dlinear1=torch.nn.Linear(128,200)
            self.dlinear2=torch.nn.Linear(200, 216)
            
            self.norm1=torch.nn.BatchNorm1d(216)
            self.norm2=torch.nn.BatchNorm1d(200)
            self.norm3=torch.nn.BatchNorm1d(128)
    
            self.relu=torch.nn.ReLU()    
        def forward(self,x):
            #encoder
            x=self.norm1(x)
            x=self.relu(self.norm2(self.linear1(x)))
            x=self.norm3(self.linear2(x))
            code=self.relu(self.norm2(self.dlinear1(x)))
            code=self.dlinear2(code)
            return x,code
    model3=autoencoder3().to(device)
    
    class SVM31(torch.nn.Module):
        def __init__(self):
            super(SVM31,self).__init__()
            self.linear1=torch.nn.Linear(128,1)
            
        def forward(self,x):
            x=self.linear1(x)
            return x
    svm31=SVM31().to(device)
    
    class SVM32(torch.nn.Module):
        def __init__(self):
            super(SVM32,self).__init__()
            self.linear1=torch.nn.Linear(128,1)
            
        def forward(self,x):
            x=self.linear1(x)
            return x
    svm32=SVM32().to(device)
    
    class autoencoder4(torch.nn.Module):
        def __init__(self):
            super(autoencoder4,self).__init__()
            #encoder
            self.linear1=torch.nn.Linear(240,200)
            self.linear2=torch.nn.Linear(200,128)
            self.linear3=torch.nn.Linear(128,64) 
            self.dlinear1=torch.nn.Linear(64, 128)
            self.dlinear2=torch.nn.Linear(128, 200)
            self.dlinear3=torch.nn.Linear(200, 240)
            
            self.norm1=torch.nn.BatchNorm1d(240)
            self.norm2=torch.nn.BatchNorm1d(200)
            self.norm3=torch.nn.BatchNorm1d(128)
            self.norm4=torch.nn.BatchNorm1d(64)
    
            self.relu=torch.nn.ReLU()    
        def forward(self,x):
            #encoder
            x=self.norm1(x)
            x=self.relu(self.norm2(self.linear1(x)))
            x=self.relu(self.norm3(self.linear2(x)))
            x=self.norm4(self.linear3(x))
            #decoder
            code=self.relu(self.norm3(self.dlinear1(x)))
            code=self.relu(self.norm2(self.dlinear2(code)))
            code=self.dlinear3(code)
            
            return x,code
    model4=autoencoder4().to(device)
    
    class SVM41(torch.nn.Module):
        def __init__(self):
            super(SVM41,self).__init__()
            self.linear1=torch.nn.Linear(64,1)
            
        def forward(self,x):
            x=self.linear1(x)
            return x
    svm41=SVM41().to(device)
    
    class SVM42(torch.nn.Module):
        def __init__(self):
            super(SVM42,self).__init__()
            self.linear1=torch.nn.Linear(64,1)
            
        def forward(self,x):
            x=self.linear1(x)
            return x
    svm42=SVM42().to(device)
    
        
    def shuffle(inputs,targets,seed):
        inp=torch.Tensor().to(device)
        tar=torch.Tensor().to(device)
        inp=inputs[seed.long()].to(device)
        tar=targets[seed.long()].squeeze().to(device)
        return inp,tar
      
    def separate(starsite,endsite,batch_size):
        setup_seed(seed_sep)
        seed=torch.randperm(batch_size)
        i,t=loadfile('data/1.csv', batch_size)
        i=i.to(device)
        t=t.to(device)
        inputs,targets=shuffle(i,t,seed)
        train1=inputs[0:starsite,:]
        test1=inputs[starsite:endsite,:]
        train1=torch.cat([train1,inputs[endsite:batch_size,:]],dim=0)
        
        i,t=loadfile('data/2.csv', batch_size)
        i=i.to(device)
        t=t.to(device)
        inputs,targets=shuffle(i,t,seed)
        train2=inputs[0:starsite,:]
        test2=inputs[starsite:endsite,:]
        train_t=targets[0:starsite]
        test_t=targets[starsite:endsite]
        train2=torch.cat([train2,inputs[endsite:batch_size,:]],dim=0)
        train_t=torch.cat([train_t,targets[endsite:batch_size]],dim=0)
        
        i,t=loadfile('data/3.csv', batch_size)
        i=i.to(device)
        t=t.to(device)
        inputs,targets=shuffle(i,t,seed)
        train3=inputs[0:starsite,:]
        test3=inputs[starsite:endsite,:]
        train3=torch.cat([train3,inputs[endsite:batch_size,:]],dim=0)
        
        i,t=loadfile('data/4.csv', batch_size)
        i=i.to(device)
        t=t.to(device)
        inputs,targets=shuffle(i,t,seed)
        train4=inputs[0:starsite,:]
        test4=inputs[starsite:endsite,:]
        train4=torch.cat([train4,inputs[endsite:batch_size,:]],dim=0)
        return train1,test1,train2,test2,train3,test3,train4,test4,train_t,test_t
    
    def lossfunction(w1,w2,y11,y12,y21,y22,x,code):
        loss=(pow(torch.norm(w1),2)+pow(torch.norm(w2),2))/2+mu1*(pow(torch.norm(y11),2)+pow(torch.norm(y22),2))+mu2*(pow(torch.norm(y11-y12-1),2)+
        pow(torch.norm(y22-y21-1),2))+mu4*torch.norm(x-code)
        return loss
    
    def change(targets,num):  #不等于一定在第一个，等于放在第二个
        targets[targets!=num]=-1
        targets[targets==num]=1
        
    def split(inputs,targets):
        inputs1=torch.Tensor().to(device)
        inputs2=torch.Tensor().to(device)
        inputs1=inputs[torch.where(targets==1)].to(device)
        inputs2=inputs[torch.where(targets==-1)].to(device)
        return inputs1,inputs2
    
    def combineloss(pre):
        loss=0
        for i in range(pre.size(1)):
            for j in range(i+1,pre.size(1)):
                loss+=torch.norm(pre[:,i]-pre[:,j])
        return loss
    
    def vote(pre,vote_num):
        temp=torch.Tensor().to(device)
        pred=torch.Tensor().to(device)
        a=torch.zeros(pre.size(0),20).to(device)
        for i in range(pre.size(0)):
            for j in range(pre.size(1)):
                a[i][int(pre[i][j])]+=1
        _,predict=torch.max(a,dim=1)
        temp=torch.cat([temp,predict],0)
        pred=torch.cat([pred,temp],0)
        for i in range(pre.size(0)):
            for j in range(pre.size(1)):
                if a[i][int(pre[i][j])]<a[i][int(temp[i])]:
                    continue
                a[i][int(pre[i][j])]+=0.5
                _,predict=torch.max(a[i],dim=0)
                temp[i]=predict
                break
        return temp
    
    def setup_seed(seed):
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          np.random.seed(seed)
          random.seed(seed)
          torch.backends.cudnn.deterministic = True
    
    def weights_init(m):
        setup_seed(seed_init)
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, pow(2/(m.weight.data.size(0)+m.weight.data.size(1)),1/2))                                             
            m.bias.data.fill_(0) 
    
    if not os.path.exists('./{}'.format(file)):
        os.mkdir('./{}'.format(file))
    
    opt_model1=torch.optim.Adam(model1.parameters(),lr=lr)
    opt_svm11=torch.optim.Adam(svm11.parameters(),lr=lr)
    opt_svm12=torch.optim.Adam(svm12.parameters(),lr=lr)
    opt_model2=torch.optim.Adam(model2.parameters(),lr=lr)
    opt_svm21=torch.optim.Adam(svm21.parameters(),lr=lr)
    opt_svm22=torch.optim.Adam(svm22.parameters(),lr=lr)
    opt_model3=torch.optim.Adam(model3.parameters(),lr=lr)
    opt_svm31=torch.optim.Adam(svm31.parameters(),lr=lr)
    opt_svm32=torch.optim.Adam(svm32.parameters(),lr=lr)
    opt_model4=torch.optim.Adam(model4.parameters(),lr=lr)
    opt_svm41=torch.optim.Adam(svm41.parameters(),lr=lr)
    opt_svm42=torch.optim.Adam(svm42.parameters(),lr=lr)
    
    
    starsite=(times-1)*scale#交叉验证起始位置
    endsite=starsite+scale#交叉验证结束位置
    train1,test1,train2,test2,train3,test3,train4,test4,train_t,test_t=separate(starsite,endsite,batch_size)
    model1.apply(weights_init),model2.apply(weights_init),model3.apply(weights_init),model4.apply(weights_init)
    svm11.apply(weights_init),svm12.apply(weights_init)
    svm21.apply(weights_init),svm22.apply(weights_init)
    svm31.apply(weights_init),svm32.apply(weights_init)
    svm41.apply(weights_init),svm42.apply(weights_init)
    for num in range(1,1+num_num):
        for epoch in range(epoch_num):
            setup_seed(randarray[k])
            k+=1
            seed=torch.randperm(len(train_t))
            i1,t1=shuffle(train1, train_t.unsqueeze(1),seed)
            i2,t2=shuffle(train2, train_t.unsqueeze(1),seed)
            i3,t3=shuffle(train3, train_t.unsqueeze(1),seed)
            i4,t4=shuffle(train4, train_t.unsqueeze(1),seed)
            for idx in range(4):
                pre1=torch.Tensor().to(device)
                pre2=torch.Tensor().to(device)
                loss=0
                for view in range(1,7):
                    if view==1:
                        inputs=i1[idx*scale:(idx+1)*scale,:].to(device)
                        targets=t1[idx*scale:(idx+1)*scale].to(device)
                        change(targets,num)
                        x1,code=model1(inputs)
                        inputs1,inputs2=split(x1, targets)
                        y_s1_pred11=svm11(inputs1).reshape(-1)
                        y_s1_pred12=svm12(inputs1).reshape(-1)
                        y_s1_pred21=svm11(inputs2).reshape(-1)
                        y_s1_pred22=svm12(inputs2).reshape(-1)
                        y1=svm11(inputs1).reshape(-1)
                        y2=svm12(inputs2).reshape(-1)
                        pre1=torch.cat([pre1,y1.unsqueeze(1)],dim=1)
                        pre2=torch.cat([pre2,y2.unsqueeze(1)],dim=1)
                        loss+=lossfunction(svm11.linear1.weight.data,svm12.linear1.weight.data,y_s1_pred11,y_s1_pred12,y_s1_pred21,y_s1_pred22,inputs,code)
                    if view==2:
                        inputs=i2[idx*scale:(idx+1)*scale,:].to(device)
                        targets=t2[idx*scale:(idx+1)*scale].to(device)
                        change(targets,num)
                        x1,code=model2(inputs)
                        inputs1,inputs2=split(x1, targets)
                        y_s2_pred11=svm21(inputs1).reshape(-1)
                        y_s2_pred12=svm22(inputs1).reshape(-1)
                        y_s2_pred21=svm21(inputs2).reshape(-1)
                        y_s2_pred22=svm22(inputs2).reshape(-1)
                        y1=svm21(inputs1).reshape(-1)
                        y2=svm22(inputs2).reshape(-1)
                        pre1=torch.cat([pre1,y1.unsqueeze(1)],dim=1)
                        pre2=torch.cat([pre2,y2.unsqueeze(1)],dim=1)
                        loss+=lossfunction(svm21.linear1.weight.data,svm22.linear1.weight.data,y_s2_pred11,y_s2_pred12,y_s2_pred21,y_s2_pred22,inputs,code)
                    if view==3:
                        inputs=i3[idx*scale:(idx+1)*scale,:].to(device)
                        targets=t3[idx*scale:(idx+1)*scale].to(device)
                        change(targets,num)
                        x1,code=model3(inputs)
                        inputs1,inputs2=split(x1, targets)
                        y_s3_pred11=svm31(inputs1).reshape(-1)
                        y_s3_pred12=svm32(inputs1).reshape(-1)
                        y_s3_pred21=svm31(inputs2).reshape(-1)
                        y_s3_pred22=svm32(inputs2).reshape(-1)
                        y1=svm31(inputs1).reshape(-1)
                        y2=svm32(inputs2).reshape(-1)
                        pre1=torch.cat([pre1,y1.unsqueeze(1)],dim=1)
                        pre2=torch.cat([pre2,y2.unsqueeze(1)],dim=1)
                        loss+=lossfunction(svm31.linear1.weight.data,svm32.linear1.weight.data,y_s3_pred11,y_s3_pred12,y_s3_pred21,y_s3_pred22,inputs,code)
                    if view==4:
                        inputs=i4[idx*scale:(idx+1)*scale,:].to(device)
                        targets=t4[idx*scale:(idx+1)*scale].to(device)
                        change(targets,num)
                        x1,code=model4(inputs)
                        inputs1,inputs2=split(x1, targets)
                        y_s4_pred11=svm41(inputs1).reshape(-1)
                        y_s4_pred12=svm42(inputs1).reshape(-1)
                        y_s4_pred21=svm41(inputs2).reshape(-1)
                        y_s4_pred22=svm42(inputs2).reshape(-1)
                        y1=svm41(inputs1).reshape(-1)
                        y2=svm42(inputs2).reshape(-1)
                        pre1=torch.cat([pre1,y1.unsqueeze(1)],dim=1)
                        pre2=torch.cat([pre2,y2.unsqueeze(1)],dim=1)
                        loss+=lossfunction(svm41.linear1.weight.data,svm42.linear1.weight.data,y_s4_pred11,y_s4_pred12,y_s4_pred21,y_s4_pred22,inputs,code)
                loss+=mu3*(combineloss(pre1)+combineloss(pre2))
                opt_model1.zero_grad()
                opt_model2.zero_grad()
                opt_model3.zero_grad()
                opt_model4.zero_grad()
                opt_svm11.zero_grad()
                opt_svm12.zero_grad()
                opt_svm21.zero_grad()
                opt_svm22.zero_grad()
                opt_svm31.zero_grad()
                opt_svm32.zero_grad()
                opt_svm41.zero_grad()
                opt_svm42.zero_grad()
                loss.backward()
                opt_model1.step()
                opt_model2.step()
                opt_model3.step()
                opt_model4.step()
                opt_svm11.step()
                opt_svm12.step()
                opt_svm21.step()
                opt_svm22.step()
                opt_svm31.step()
                opt_svm32.step()
                opt_svm41.step()
                opt_svm42.step()
            print(f"time:{times} num:{num} eopch:{epoch+1}/{epoch_num}: loss:{loss}")
        torch.save(model1.state_dict(),'{}/model1{}.pth'.format(file,num))
        torch.save(svm11.state_dict(),'{}/svm11{}.pth'.format(file,num))
        torch.save(svm12.state_dict(),'{}/svm12{}.pth'.format(file,num))
        torch.save(model2.state_dict(),'{}/model2{}.pth'.format(file,num))
        torch.save(svm21.state_dict(),'{}/svm21{}.pth'.format(file,num))
        torch.save(svm22.state_dict(),'{}/svm22{}.pth'.format(file,num))
        torch.save(model3.state_dict(),'{}/model3{}.pth'.format(file,num))
        torch.save(svm31.state_dict(),'{}/svm31{}.pth'.format(file,num))
        torch.save(svm32.state_dict(),'{}/svm32{}.pth'.format(file,num))
        torch.save(model4.state_dict(),'{}/model4{}.pth'.format(file,num))
        torch.save(svm41.state_dict(),'{}/svm41{}.pth'.format(file,num))
        torch.save(svm42.state_dict(),'{}/svm42{}.pth'.format(file,num))
        
    cor,tol,cor1,tol1,cor2,tol2,cor3,tol3,cor4,tol4=0,0,0,0,0,0,0,0,0,0
    pre=torch.Tensor().to(device)
    for idx in range(1):
        for view in range(1,7):
            temppre=torch.Tensor().to(device)
            if view==4:
                inputs=test1[idx*scale:(idx+1)*scale,:].to(device)
                targets=test_t[idx*scale:(idx+1)*scale].to(device)
                for num in range(1,1+num_num):
                    model=autoencoder1()
                    model.load_state_dict(torch.load('{}/model1{}.pth'.format(file,num)))
                    model=model.to(device)
                    svm1=SVM11()
                    svm1.load_state_dict(torch.load('{}/svm11{}.pth'.format(file,num)))
                    svm1=svm1.to(device)
                    svm2=SVM12()
                    svm2.load_state_dict(torch.load('{}/svm12{}.pth'.format(file,num)))
                    svm2=svm2.to(device)
                    x,c=model(inputs)
                    y1=svm1(x)
                    temppre=torch.cat([temppre,y1],1)
                _,predict=torch.min(abs(temppre),dim=1)
                predict=predict+1
                cor1+=(predict==targets).sum().item()
                tol1+=len(predict)
                acc1=cor1*100/tol1
                print("acc1:",acc1)
                pre=torch.cat([pre,predict.unsqueeze(1)],1)
            if view==3:
                inputs=test2[idx*scale:(idx+1)*scale,:].to(device)
                targets=test_t[idx*scale:(idx+1)*scale].to(device)
                for num in range(1,1+num_num):
                    model=autoencoder2()
                    model.load_state_dict(torch.load('{}/model2{}.pth'.format(file,num)))
                    model=model.to(device)
                    svm1=SVM21()
                    svm1.load_state_dict(torch.load('{}/svm21{}.pth'.format(file,num)))
                    svm1=svm1.to(device)
                    svm2=SVM22()
                    svm2.load_state_dict(torch.load('{}/svm22{}.pth'.format(file,num)))
                    svm2=svm2.to(device)
                    x,c=model(inputs)
                    y1=svm1(x)
                    temppre=torch.cat([temppre,y1],1)
                _,predict=torch.min(abs(temppre),dim=1)
                predict=predict+1
                cor2+=(predict==targets).sum().item()
                tol2+=len(predict)
                acc2=cor2*100/tol2
                print("acc2:",acc2)
                pre=torch.cat([pre,predict.unsqueeze(1)],1)
            if view==1:
                inputs=test3[idx*scale:(idx+1)*scale,:].to(device)
                targets=test_t[idx*scale:(idx+1)*scale].to(device)
                for num in range(1,1+num_num):
                    model=autoencoder3()
                    model.load_state_dict(torch.load('{}/model3{}.pth'.format(file,num)))
                    model=model.to(device)
                    svm1=SVM31()
                    svm1.load_state_dict(torch.load('{}/svm31{}.pth'.format(file,num)))
                    svm1=svm1.to(device)
                    svm2=SVM32()
                    svm2.load_state_dict(torch.load('{}/svm32{}.pth'.format(file,num)))
                    svm2=svm2.to(device)
                    x,c=model(inputs)
                    y1=svm1(x)
                    temppre=torch.cat([temppre,y1],1)
                _,predict=torch.min(abs(temppre),dim=1)
                predict=predict+1
                cor3+=(predict==targets).sum().item()
                tol3+=len(predict)
                acc3=cor3*100/tol3
                print("acc3:",acc3)
                pre=torch.cat([pre,predict.unsqueeze(1)],1)
            if view==2:
                inputs=test4[idx*scale:(idx+1)*scale,:].to(device)
                targets=test_t[idx*scale:(idx+1)*scale].to(device)
                for num in range(1,1+num_num):
                    model=autoencoder4()
                    model.load_state_dict(torch.load('{}/model4{}.pth'.format(file,num)))
                    model=model.to(device)
                    svm1=SVM41()
                    svm1.load_state_dict(torch.load('{}/svm41{}.pth'.format(file,num)))
                    svm1=svm1.to(device)
                    svm2=SVM42()
                    svm2.load_state_dict(torch.load('{}/svm42{}.pth'.format(file,num)))
                    svm2=svm2.to(device)
                    x,c=model(inputs)
                    y1=svm1(x)
                    temppre=torch.cat([temppre,y1],1)
                _,predict=torch.min(abs(temppre),dim=1)
                predict=predict+1
                cor4+=(predict==targets).sum().item()
                tol4+=len(predict)
                acc4=cor4*100/tol4
                print("acc4:",acc4)
                pre=torch.cat([pre,predict.unsqueeze(1)],1)
        predict=vote(pre,vote_num)
        cor+=(predict==targets).sum().item()
        tol+=len(targets)
        acc=cor*100/tol
        print('acc:',acc)
    return acc        

if __name__ == '__main__':
    for times in range(1,6):
        torch.cuda.synchronize()
        starttime = time.time()
        device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
        batch_size=2000#样本总大小
        scale=int(batch_size/5)#每组样本的大小
        DMVTSVM(0.005,1,1,pow(2,-2),8,80,30,10,0,batch_size,scale,'model2',20,1000,[1,1,1,1],times)
        endtime = time.time()
        print('time:',endtime-starttime)


































































