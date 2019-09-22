import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb

def custom_callback():
   def callback(env):
       print('callback:', env.evaluation_result_list)
   return callback

dtrain = xgb.DMatrix("agaricus.txt.train")
dtest = xgb.DMatrix("agaricus.txt.test")

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 5

#Train the model
bst = xgb.train(param, dtrain, num_round, watchlist,callbacks=[custom_callback()])

preds = bst.predict(dtest)
labels = dtest.get_label()
print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
bst.save_model('0001.model')

# dump model
bst.dump_model('dump.raw.txt')

# save dmatrix into binary buffer
dtest.save_binary('dtest.buffer')

# save model
bst.save_model('xgb.model')

# load model and data
bst2 = xgb.Booster(model_file='xgb.model')
dtest2 = xgb.DMatrix('dtest.buffer')
preds2 = bst2.predict(dtest2)

# assert they are the same
assert np.sum(np.abs(preds2 - preds)) == 0

# alternatively, you can pickle the booster
pks = pickle.dumps(bst2)

# load model and data in
bst3 = pickle.loads(pks)
preds3 = bst3.predict(dtest2)

# assert they are the same
assert np.sum(np.abs(preds3 - preds)) == 0

#Plot the important feature
xgb.plot_importance(bst)
