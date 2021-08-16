import torch
def test_out(model,test_loader,criterion,inv_map,device):
  with torch.no_grad():
    model.eval()
    prediction=[]
    for batch, data in enumerate(test_loader):
      data=data.to(device)
      output=model(data.float())
      predicted=torch.argmax(output.data,1)
      print(predicted,'predicted')
      prediction.append(int(inv_map[predicted.item()]))
    return prediction