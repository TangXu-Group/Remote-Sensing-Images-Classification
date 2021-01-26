from model import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 64
learn_rate = 1e-4
num_classes = 21
num_epochs = 300
keep_rate = 0.5

tf.set_random_seed(1)


TRAIN_PATH = ".../train.txt"  
TEST_PATH = ".../test.txt"


train_filenames, train_filelabels,test_filenames, test_filelabels = read_txt(TRAIN_PATH,TEST_PATH)

print("training data size {}, training label size {}".format(train_filenames.shape, train_filelabels.shape))
print("testing data size {}, testing label size {}".format(test_filenames.shape, test_filelabels.shape))

x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="x")
y = tf.placeholder(tf.float32, [None, num_classes], name="y")
B = tf.placeholder(tf.bool,name = "train_bn")
kp = tf.placeholder(tf.float32, [], name="kp")
lr= tf.placeholder(tf.float32, [], name="learn_rate")


feature_0,feature_1,mask_SE,mask_SP = VGG16(x,weights,B)


mask_SE_0,mask_SE_1 = tf.split(mask_SE,2,0,"split_pool5_SE")
mask_SP_0,mask_SP_1 = tf.split(mask_SP,2,0,"split_pool5_SP")

label_logits = fc(feature_0,feature_1,weights,kp)
label_logits_0,label_logits_180 = tf.split(label_logits,2,0,"label_logits")



# loss
with tf.name_scope('loss'):

    cross_entropy_0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=(label_logits_0)))
    cross_entropy_180 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=(label_logits_180)))
    loss_diffierent = tf.reduce_mean(tf.square(mask_SE_0-tf.image.rot90(mask_SE_1,2))) + tf.reduce_mean(tf.square(mask_SP_0-tf.image.rot90(mask_SP_1,2)))




    weights_decays = (tf.nn.l2_loss(weights["conv_weights1_1"]) +
                      tf.nn.l2_loss(weights["conv_weights1_2"]) +
                      tf.nn.l2_loss(weights["conv_weights2_1"]) +
                      tf.nn.l2_loss(weights["conv_weights2_2"]) +
                      tf.nn.l2_loss(weights["conv_weights3_1"]) +
                      tf.nn.l2_loss(weights["conv_weights3_2"]) +
                      tf.nn.l2_loss(weights["conv_weights3_3"]) +
                      tf.nn.l2_loss(weights["conv_weights4_1"]) +
                      tf.nn.l2_loss(weights["conv_weights4_2"]) +
                      tf.nn.l2_loss(weights["conv_weights4_3"]) +
                      tf.nn.l2_loss(weights["conv_weights5_1"]) +
                      tf.nn.l2_loss(weights["conv_weights5_2"]) +
                      tf.nn.l2_loss(weights["conv_weights5_3"])
                      )

    loss_op = cross_entropy_0 + cross_entropy_180 + 5e-3 * weights_decays + loss_diffierent


with tf.name_scope('optimizer'):
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_op)

# accuracy
with tf.name_scope("accuracy"):

    correct_pred_0 = tf.equal(tf.argmax(tf.nn.softmax(label_logits_0), 1), tf.argmax(y, 1))
    correct_pred_180 = tf.equal(tf.argmax(tf.nn.softmax(label_logits_180), 1), tf.argmax(y, 1))

    correct_pred = tf.concat([correct_pred_0, correct_pred_180], axis=0)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre_index = 0
    pre = []
    real = []

    acc_list = [0.0, 0.0]

    for epoch in range(1, num_epochs + 1):
        index_train = np.arange(train_filenames.shape[0])
        np.random.shuffle(index_train)
        current_train_filenames = train_filenames[index_train, ...]
        current_train_filelabels = train_filelabels[index_train, ...]
        if epoch == 10:
            learn_rate = learn_rate / 10.0  
        if epoch == 30:
            learn_rate = learn_rate / 10.0

        for batch in range((current_train_filenames.shape[0] // batch_size)-1):
            start = batch * batch_size
            end = min(start + batch_size, current_train_filenames.shape[0])

            current_traindata_0 = read_data(current_train_filenames[start:end])
            current_traindata_180 = read_data_180(current_train_filenames[start:end])
            current_traindata = (current_traindata_0+current_traindata_180)/2.0
            current_trainlabel = np.eye(num_classes)[current_train_filelabels[start:end]]

            train_feed_dict = {x: current_traindata, y: current_trainlabel, kp: keep_rate,lr: learn_rate,B:True}
            _, batch_loss, batch_acc   = sess.run([train_op,loss_op, accuracy], feed_dict=train_feed_dict)

            line = "epoch: %d/%d, start:%d ,end:%d,train_loss: %.4f,trian_acc: %.4f,max_test_acc :%.4f\n" % (
                epoch, num_epochs, start, end, batch_loss, batch_acc, max(acc_list))
            print(line)


        if epoch > 10:
            for i in range(6):
                start_test = 70 * i
                end_test = min(start_test + 70, test_filenames.shape[0])
                current_testdata_0 = read_data(test_filenames[start_test:end_test])
                current_testdata_180 = read_data_180(test_filenames[start_test:end_test])
                current_testdata = (current_testdata_0+current_testdata_180)/2.0

                current_testlabel = np.eye(num_classes)[test_filelabels[start_test:end_test]]

                test_feed_dict = {x: current_testdata, y: current_testlabel, kp: 1.0, B: False}

                pre_val = sess.run([tf.argmax(tf.nn.softmax((label_logits_0 + label_logits_180)/2.0), 1)],
                                   feed_dict=test_feed_dict)
                pre.append(pre_val)

            pre = np.reshape(np.array(pre), newshape=(420,))
            real = np.array(test_filelabels)

            test_acc = np.mean(np.array(np.equal(pre, real), dtype=bool))
            print(test_acc)

            pre = []

            if epoch > 0 and test_acc > max(acc_list):
                acc_list.append(test_acc)
                acc_list = acc_list[1:]

                saver = tf.train.Saver()
                saver.save(sess, "./checkpoint_dir_new/MyModel.ckpt")
                print("acc_list is {0}   and model save  OK".format(acc_list))





