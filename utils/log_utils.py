import os
import csv

def prepare_csv(log_path,dataset,model,depth,optimizer,noise_rate,damping,experiment_type):
    if model in ['vit_small','vit_base','vit_large','vit_huge']:
        train_log_path = log_path+'/'+experiment_type+'/'+dataset+'/'+model+'/train'
        test_log_path = log_path+'/'+experiment_type+'/'+dataset+'/'+model+'/test'
    else:
        train_log_path = log_path+'/'+experiment_type+'/'+dataset+'/'+model+str(depth)+'/train'
        test_log_path = log_path+'/'+experiment_type+'/'+dataset+'/'+model+str(depth)+'/test'
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
    if not os.path.exists(test_log_path):
        os.makedirs(test_log_path)
    if experiment_type == 'error' and noise_rate==0.0:
        csv_train = open(train_log_path+'/'+optimizer.lower()+'.csv','a+',newline='')
        csv_test = open(test_log_path+'/'+optimizer.lower()+'.csv','a+',newline='')
    elif experiment_type == 'damping' and noise_rate==0.0:
        csv_train = open(train_log_path+'/'+optimizer.lower()+'_damping'+str(damping)+'.csv','a+',newline='')
        csv_test = open(test_log_path+'/'+optimizer.lower()+'_damping'+str(damping)+'.csv','a+',newline='')
    else:
        csv_train = open(train_log_path+'/'+optimizer.lower()+'_noise'+str(noise_rate)+'.csv','a+',newline='')
        csv_test = open(test_log_path+'/'+optimizer.lower()+'_noise'+str(noise_rate)+'.csv','a+',newline='')
    csv_train_writer = csv.writer(csv_train)
    csv_test_writer = csv.writer(csv_test)
    return csv_train,csv_train_writer,csv_test,csv_test_writer

def write_csv(csv_train,csv_train_writer,csv_test,csv_test_writer,head=False,train=False,test=False,args=None,**kwargs):
    if head:
        if args is None:
            raise ValueError('Parser is None! Please check!')
        csv_train_writer.writerow([
                'network:',args.network,'depth',args.depth,'Loss:CrossEntropy','Dataset:',args.dataset,'Optimizer:',kwargs['optim_name'],'LearningRate:',
        args.learning_rate,'BatchSize:',args.batch_size,'EpochRange:',args.epoch])
        csv_train_writer.writerow(['Epoch','Train_Loss','Train_Accuracy','Train_Time'])
        csv_train.flush()
        csv_test_writer.writerow([
            'network:',args.network,'depth',args.depth,'Loss:CrossEntropy','Dataset:',args.dataset,'Optimizer:',kwargs['optim_name'],'LearningRate:',
            args.learning_rate,'BatchSize:',args.batch_size,'EpochRange:',args.epoch])
        csv_test_writer.writerow(['Epoch','Test_Loss','Test_Accuracy','Generalization_Gap'])
        csv_test.flush()
    elif train:
        csv_train_writer.writerow([kwargs['epoch']+1,
                                   kwargs['train_loss'] / (kwargs['batch_index'] + 1),
                                   100.*kwargs['correct'] / kwargs['total'],
                                   round(kwargs['time_elapsed']/60,1)])
        csv_train.flush()
    elif test:
        csv_test_writer.writerow([kwargs['epoch']+1,
                                  kwargs['test_loss'] / (kwargs['batch_index']+1),
                                  kwargs['acc'],
                                  abs((kwargs['test_loss'] / (kwargs['batch_index']+1)) - (kwargs['train_loss']))
        ])
        csv_test.flush()
    else:
        raise ValueError('head,train,test are all False! Nothing to write!')