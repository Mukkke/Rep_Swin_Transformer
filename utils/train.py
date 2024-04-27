# +
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils.build_model import SwinModel
from utils.build_loader import create_dataset, create_test_dataset
from utils.get_config import get_config
from utils.logger import create_logger
from utils.cos_lr_warmup import cosine_warmup_schedule
import time
import argparse
import os
print("Current working directory:", os.getcwd())

def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('--train_data', type=str, default='imagenet/train', help='Path to the training data.')
    parser.add_argument('--test_data', type=str, default='imagenet/val', help='Path to the test data.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config YAML file.')
    parser.add_argument('--record_steps', type=int, default=40, help='Number of record steps.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size.')
    args, _ = parser.parse_known_args()
    print(args.config)
    config = get_config(args.config)
    return args, config



def train(args, config):
    tf.random.set_seed(42)
    output_directory = os.getcwd()
    config = get_config()
    model = SwinModel(config)
    train_data_path = args.train_data
    test_data_path = args.test_data
    batch_size = args.batch_size
    record_steps = args.record_steps
    epochs = config['TRAIN']['EPOCHS']
    early_stop_patience = 3
    train_dataset, val_dataset = create_dataset(batch_size, train_data_path)
    
    logger = create_logger(output_directory, dist_rank=0, name='swim_logger')
    
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    
    optimizer = tf.optimizers.Adam(learning_rate=float(config['TRAIN']['WARMUP_LR']))
    
    train_loss = tf.keras.metrics.Mean(name = 'train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'train_accuracy')
    
    val_loss = tf.keras.metrics.Mean(name = 'val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'val_accuracy')
    
    best_val_accuracy = 0.0
    best_model_epoch = 0
    consecutive_epochs_without_improvement = 0

    for epoch in range(epochs):
        train_dataset, val_dataset = create_dataset(batch_size)
        lr_schedule = cosine_warmup_schedule(epoch, config)
        optimizer.learning_rate.assign(lr_schedule)

        logger.info(f"\nStart of epoch {epoch+1}")
        start_train_time = time.time()
        
        epoch_loss = 0
        for step, (images_batch_train, labels_batch_train) in enumerate(train_dataset):
            ### train_step
            with tf.GradientTape() as tape:
                predict_labels = model(images_batch_train, training = True)
                loss_value = loss_object(labels_batch_train, predict_labels)
                
            
            gradients = tape.gradient(loss_value, model.trainable_variables)
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_accuracy.update_state(labels_batch_train, predict_labels)
            train_loss.update_state(loss_value)
            
            epoch_loss+= float(loss_value)
            train_loss_result = train_loss.result().numpy()
            train_accuracy_result = train_accuracy.result().numpy()
            
            if step % record_steps == 0:
                logger.info(f"{epoch+1} epoch {step+1} step: Training loss (for one batch): {train_loss_result:.4f},Training accuracy: {train_accuracy_result:.4f}")
        
        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        train_loss_result = train_loss.result()
        train_accuracy_result = train_accuracy.result()
        train_loss.reset_states()
        train_accuracy.reset_states()

        avg_epoch_loss = epoch_loss / len(train_dataset)
        logger.info(f"Epoch {epoch+1}, Training Loss: {avg_epoch_loss}")
        logger.info(f"{epoch+1} epoch: Training loss: {train_loss_result:.4f}, Training accuracy: {train_accuracy_result:.4f}")
        logger.info(f"{epoch+1} epoch: Training time: {train_time:.2f} seconds.")
        
        start_val_time = time.time()
        for val_step, (images_batch_val, labels_batch_val) in enumerate(val_dataset):
            ###validation_step
            predict_val = model(images_batch_val, training = False)
            loss_value = loss_object(labels_batch_val, predict_val)
            val_loss.update_state(loss_value)
            val_accuracy.update_state(labels_batch_val, predict_val)
            
        end_val_time = time.time()
        val_time = end_val_time - start_val_time
        val_loss_result = val_loss.result().numpy()
        val_accuracy_result = val_accuracy.result().numpy()
        val_loss.reset_states()
        val_accuracy.reset_states()

        logger.info(f"{epoch+1} epoch: Validation loss: {val_loss_result:.4f}, Valiadation accuracy: {val_accuracy_result:.4f}")
        logger.info(f"{epoch+1} epoch: Validation time: {val_time:.2f} seconds.")
        
            
        if val_accuracy_result > best_val_accuracy:
            best_val_accuracy = val_accuracy_result
            consecutive_epochs_without_improvement = 0
            best_model_epoch = epoch + 1
            save_name = f"./savemodel/model_epoch{best_model_epoch}_valacc{best_val_accuracy:.2f}.ckpt"
            model.save_weights(save_name, save_format="tf")
            logger.info(f"Saved the best model at: {save_name}")
        else:
            consecutive_epochs_without_improvement += 1
            
        if consecutive_epochs_without_improvement > early_stop_patience:
            logger.info(f"Early stopping after {epoch + 1} epochs without improvement.")
            break
    logger.info("Training completed.")

'''
best_model = SwinModel(config)  
best_model.load_weights(f"./savemodel/model_epoch{best_model_epoch}_valacc{best_val_accuracy:.2f}.ckpt")

test_dataset = create_test_dataset(batch_size=batch_size)
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.CategoricalAccuracy()

for images_batch_test, labels_batch_test in test_dataset:
    predict_test = best_model(images_batch_test, training=False)
    loss_value_test = loss_object(labels_batch_test, predict_test)
    test_loss.update_state(loss_value_test)
    test_accuracy.update_state(labels_batch_test, predict_test)

test_loss_result = test_loss.result().numpy()
test_accuracy_result = test_accuracy.result().numpy()
logger.info(f"Test loss: {test_loss_result:.4f}, Test accuracy: {test_accuracy_result:.4f}")
'''

if __name__ == "__main__":
    args, config, _ = parse_arguments()
    train(args, config)
