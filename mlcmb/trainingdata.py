import numpy as np
import quicklens as ql
import tensorflow as tf
import sys
from shutil import copyfile

import config
import utilities

#run e.g.: python trainingdata.py ~/mlcmb_secondaries/mlcmb/config_master.ini

##################### make data set
#creates tfrecords for training and validation, and a numpy array for test


def make_dataset(configpath):
    
    params = config.Parameters(configpath)

    datasetid = params.datasetid

    nsims_train = params.nsims_train
    nsims_valid = params.nsims_valid
    nsims_test = params.nsims_test
    nsims      = nsims_train+nsims_valid+nsims_test

    #for ell space computations
    lmax       = params.lmax                                 # maximum multipole.
    nside = params.nx
    nx         = params.nx
    dx         = params.dx

    #noise levels 
    nlev_t     = params.nlev_t                                 # 10. temperature map noise level, in uK.arcmin.
    nlev_p     = params.nlev_p                                  # 10. polarization map noise level (Q, U), in uK.arcmin.
    bl         = ql.spec.bl(fwhm_arcmin=params.fwhm_arcmin, lmax=lmax) # instrumental beam transfer function.

    cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)  # cmb theory spectra.

    mask = params.mask

    nltt       = (nlev_t*np.pi/180./60.)**2 / bl**2
    nlee       = (nlev_p*np.pi/180./60.)**2 / bl**2

    #make a TFrecord for training set and valid set, but not for test set
    filename_train = params.datapath+"datasets/dataset_train_"+str(datasetid)+".tfrecords"
    filename_valid = params.datapath+"datasets/dataset_valid_"+str(datasetid)+".tfrecords"
    print('Writing', filename_train,filename_valid)

    def _bytes_feature_image(image):
        value = tf.compat.as_bytes(image.tostring())
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    
    dataset_test = np.zeros( (nsims_test,nside,nside,16) ) 

    with tf.python_io.TFRecordWriter(filename_train) as writer_train, tf.python_io.TFRecordWriter(filename_valid) as writer_valid:

        for map_id in range(nsims):
            print ("map", map_id)

            tcmb =  #T cmb
            rhogal =  #T obs
            vrad =  #E sky        

            #-----save record and test data

            #write records
            example = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'tcmb': _bytes_feature_image(tcmb),
                      'rhogal': _bytes_feature_image(rhogal),
                      'vrad': _bytes_feature_image(vrad)
                  }))
            
            if map_id<nsims_train:
                writer_train.write(example.SerializeToString())
            if map_id>=nsims_train and map_id<(nsims_train+nsims_valid):
                writer_valid.write(example.SerializeToString()) 

            #save test data
            if map_id>=nsims_train+nsims_valid:
                testmap_id=map_id-nsims_valid-nsims_train

                dataset_test[testmap_id,:,:,0] = tcmb
                dataset_test[testmap_id,:,:,1] = rhogal
                dataset_test[testmap_id,:,:,2] = vrad
       
    #save training set
    if nsims_test>0:
        np.save(params.datapath+"datasets/dataset_test_"+str(datasetid)+".npy",dataset_test)

    #save config for this data set
    copyfile(configpath, params.datapath+"datasets/dataset_config_backup_"+str(datasetid)+".ini") 
    
        
        
        
        
        
        
##################### TF record parser

#https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36
#https://www.tensorflow.org/guide/datasets#parsing_tfexample_protocol_buffer_messages
#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb
#Many input pipelines extract tf.train.Example protocol buffer messages from a TFRecord-format file (written, for example, using tf.python_io.TFRecordWriter). Each tf.train.Example record contains one or more "features", and the input pipeline typically converts these features into tensors.

def tfrecord_parse_function(proto,npad,params):
    if params.wf_mode == "T":
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'tcmb': tf.FixedLenFeature([], tf.string),
                            'rhogal': tf.FixedLenFeature([], tf.string),
                            'vrad': tf.FixedLenFeature([], tf.string)}

        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        # Turn your saved image string into an array
        parsed_features['tcmb'] = tf.decode_raw(parsed_features['tcmb'], tf.float64)*params.map_rescale_factor
        parsed_features['rhogal'] = tf.decode_raw(parsed_features['rhogal'], tf.float64)*params.map_rescale_factor
        parsed_features['vrad'] = tf.decode_raw(parsed_features['vrad'], tf.float64)
        
        #reshape to original form
        parsed_features['tcmb'] = tf.reshape(parsed_features['tcmb'], [params.nx, params.nx, 1])
        parsed_features['rhogal'] = tf.reshape(parsed_features['rhogal'], [params.nx, params.nx, 1])
        parsed_features['vrad'] = tf.reshape(parsed_features['vrad'], [params.nx, params.nx, 1])
        
        #pad
        #_padfunc = utilities.periodic_padding
        #parsed_features['tcmb_pad'] = tf.py_func(_padfunc, [parsed_features['tcmb'],npad], tf.float64 )
        #parsed_features['rhogal_pad'] = tf.py_func(_padfunc, [parsed_features['rhogal'],npad], tf.float64 )
        #parsed_features['vrad_pad'] = tf.py_func(_padfunc, [parsed_features['vrad'],npad], tf.float64 )

        image = tf.concat([parsed_features['tcmb'],parsed_features['rhogal']],axis=-1)
        label = parsed_features['vrad']
        return image, label
        
        
        
       
 
def main():
    configpath=sys.argv[1]
    make_dataset(configpath)
        
        
if __name__ == "__main__":
    main()       
        
        