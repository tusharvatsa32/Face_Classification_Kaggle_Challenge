import time
import torch
def train_epoch(model, train_loader, criterion, optimizer,device):
    model.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device

        total_predictions += target.size(0)
        
        outputs = model(data.float())
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == target).sum().item()
        loss = criterion(outputs, target.long())
        running_loss += loss.item()
        acc = (correct_predictions/total_predictions)*100.0
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    print('Training acc:',acc,'Time:',end_time-start_time,'s')
    return running_loss



def val_epoch(model, test_loader, criterion,device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.to(device)

            outputs = model(data.float())

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target.long()).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc

