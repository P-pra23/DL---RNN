# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset



## DESIGN STEPS
## STEP 1:

Import the required libraries and load the historical stock price dataset.

## STEP 2:

Preprocess the stock data by selecting closing prices, normalizing values, and creating sequences.

## STEP 3:

Split the dataset into training and testing sets for model evaluation.

## STEP 4:

Build the RNN model with recurrent layers and dense output layers.

## STEP 5:

Train the RNN model using historical stock price sequences.

## STEP 6:

Test the trained model and compare predicted stock prices with actual prices.





## PROGRAM

### Name:Prathikshaa

### Register Number:212224100043

```python
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out




model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
      total_loss=0
      for x_batch,y_batch in train_loader:
        x_batch,y_batch=x_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        outputs=model(x_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss/len(train_loader))
      print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
      # Plot training loss
    print('Name: Prathikshaa')
    print('Register Number: 212224100043')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
train_model(model,train_loader,criterion,optimizer)


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="1280" height="645" alt="image" src="https://github.com/user-attachments/assets/4aa8ad78-9325-418a-aada-0b90b5d68f9c" />


## True Stock Price, Predicted Stock Price vs time

<img width="1475" height="717" alt="image" src="https://github.com/user-attachments/assets/156739be-68fc-4ef0-9e7f-385cfee891d7" />


### Predictions
<img width="448" height="88" alt="image" src="https://github.com/user-attachments/assets/f2a56169-afc7-4a5c-931a-b41b5972ab16" />



## RESULT

The Recurrent Neural Network (RNN) model was successfully developed and trained using historical stock price data. The model learned the sequential patterns in stock prices and accurately predicted future closing prices based on past trends. The predicted values closely matched the actual stock prices, demonstrating the effectiveness of RNNs in time-series forecasting and stock market prediction applications.
