from keras import models

print 'loading architecture and weights...'
model = models.model_from_json(open('model.json').read())
model.load_weights('weights.h5')

print 'loading dataset...'
