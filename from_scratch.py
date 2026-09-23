import pandas as pd
import numpy as np

df = pd.read_csv('forestfires.csv')
print(df.columns)
print(df['area'].head(-5))
print(df.info())

X = df.drop(columns=['day','month','area'])
y = df['area']

np.random.seed(42)

indices = np.random.permutation(len(X))
test_size = int(len(X)*0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

y_train_log = np.log1p(y_train)
def weight(g,h,l):
  G = sum(g)
  H = sum(h)
  w = -G/(H+l)
  return w

def find_thresholds(x):
  thresholds = []
  for feature_index in range(len(x[0])):
    feature_values = []
    for i in range(len(x)):
      feature_values.append(x[i][feature_index])
    thres = []
    for i in range(len(feature_values)-1):
      thres.append((feature_values[i]+feature_values[i+1])/2)
    thresholds.append(thres)
  return thresholds

def split_data(x,g,h,feature_index,threshold):
   x_right = []
   x_left = []
   g_right = []
   g_left = []
   h_right = []
   h_left = []
   for i in range(len(x)):
    if x[i][feature_index] <= threshold:
       x_left.append(x[i])
       g_left.append(g[i])
       h_left.append(h[i])
    else:
       x_right.append(x[i])
       g_right.append(g[i])
       h_right.append(h[i])
   return x_left,x_right,g_left,g_right,h_left,h_right

def find_gain(y,g_left,g_right,h_left,h_right,l,p):
   pred = np.mean(y)
   left_g = np.sum(g_left)
   right_g = np.sum(g_right)
   left_h = np.sum(h_left)
   right_h = np.sum(h_right)
   parent_g = np.sum(pred - y)
   parent_h = np.ones(len(y))
   gain = 0.5*(((left_g)**2/(left_h+l) + (right_g)**2/(right_h+l))-((parent_g)**2/(parent_h+l))) - p
   return np.max(gain)

def best_split(x,y,g,h,l,p,threholds):
   best_gain = float('-inf')
   best_feature = 0
   best_threshold = 0
   best_x_left = None
   best_x_right = None
   best_g_left = None
   best_g_right = None
   best_h_left = None
   best_h_right = None
   for feature_index in range(len(x[0])):
      for threshold in threholds[feature_index]:
        x_left,x_right,g_left,g_right,h_left,h_right = split_data(x,g,h,feature_index,threshold)
        if len(x_left) == 0 or len(x_right) == 0:
           continue
        gain = find_gain(y,g_left,g_right,h_left,h_right,l,p)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_index
            best_threshold = threshold
            best_x_left = x_left
            best_x_right = x_right
            best_g_left = g_left
            best_g_right = g_right
            best_h_left = h_left
            best_h_right = h_right
   return best_gain,best_feature,best_threshold,best_x_left,best_x_right,best_g_left,best_g_right,best_h_left,best_h_right

def build_tree(x,y,g,h,l,p,depth=0,max_depth=4):
   if depth >= max_depth:
      return weight(g,h,l)
   thresholds = find_thresholds(x)
   best_gain,best_feature,best_threshold,best_x_left,best_x_right,best_g_left,best_g_right,best_h_left,best_h_right = best_split(x,y,g,h,l,p,thresholds)
   if best_gain <= 0:
      return weight(g,h,l)
   if best_g_left is None:
      return weight(g,h,l)
   left_subtree = build_tree(best_x_left,y,best_g_left,best_h_left,l,p,depth+1,max_depth)
   right_subtree = build_tree(best_x_right,y,best_g_right,best_h_right,l,p,depth+1,max_depth)
   return best_feature,best_threshold,left_subtree,right_subtree

def predict(sample,tree):
   if isinstance(tree,(float,np.floating)):
      return tree
   best_feature,best_threshold,left_subtree,right_subtree = tree
   if sample[best_feature] <= best_threshold:
      return predict(sample,left_subtree)
   else:
      return predict(sample,right_subtree)

def prediction(x_test,tree):
   predictions = []
   for sample in x_test:
      predictions.append(predict(sample,tree))
   return predictions 

lr = 0.1
def xgboost(x,y,l=1,p=1):
   pred = np.mean(y)
   g = pred - y
   h = np.ones(len(y))
   trees = []
   for _ in range(100):
      g = pred - y
      h = np.ones(len(y))
      tree = build_tree(x,y,g,h,l,p)
      tree_pred = np.array(prediction(x,tree))
      pred += lr*tree_pred
      trees.append(tree)
   return trees

def test_pred(x_test,x,y_train):
   pred = np.mean(y_train)
   trees = xgboost(x,y_train)
   for tree in trees:
      tree_pred = np.array(prediction(x_test,tree))
      pred += lr*tree_pred
   return pred
pred_log = test_pred(x_test,x_train,y_train_log)
pred = np.expm1(pred_log)
mse = np.mean((pred - y_test)**2)
mae = np.mean(abs(pred - y_test))
rmse = np.sqrt(mse)
ss_total = np.sum((y_test - np.mean(y_test))**2)
ss_residual = np.sum((y_test - pred)**2)
r2 = 1 - (ss_residual / ss_total)

print('MAE:',mae)
print('MSE:',mse)
print('RMSE:',rmse)
print('R2 Score:',r2)