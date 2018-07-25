
#ifdef __cplusplus
extern "C"{
#endif //__cplusplus

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "checkpointing.h"
#include "parson/parson.h"

extern char* __progname;

void checkpoint_write(
    const char* filename,
    const checkpoint_t* checkpoint
){
    assert(filename);
    assert(checkpoint);

    FILE* fp =  fopen(filename, "wb+");
    if(fp!= NULL){
        if(checkpoint->size > 0){
            fwrite(checkpoint->stream, sizeof(byte), checkpoint->size, fp);
        }
        fsync(fp);
    }
    fclose(fp);
}

void checkpoint_read(
    const char* filename,
    checkpoint_t* checkpoint
){
    assert(checkpoint);
    assert(filename); 

    byte* ptr = checkpoint->stream;
    int count = 0;
    FILE* fp =  fopen(filename, "rb");
    
    if(fp!= NULL){
        if(checkpoint->size > 0){
            while(count < checkpoint->size          //prevent buffer overrun
                && (fread(ptr, 1, 1, fp) == 1)){    //read until EOF
                    ptr++;
                    count++;
            }
            checkpoint->size = count;
        }
    }
}


void checkpointnow(
    const int n_files, 
    const char** filename,
    const checkpoint_t** data    
){
    assert(filename);
    assert(data);
    
    int i;
    char chkptfilename[64];
    FILE* fp;
    char* json_string;

    if(n_files>0){

        for(i=0; i<n_files; i++){
            checkpoint_write(filename[i], data[i]);
        }

        snprintf(chkptfilename, 64, "%s.chkpt.json", __progname);

        JSON_Value* root_value = json_value_init_object();
        JSON_Object* root_object = json_value_get_object(root_value);
        json_object_set_string(root_object, "program", __progname);
        json_object_set_number(root_object, "filecount", n_files);
        JSON_Value* files = json_value_init_array();
        JSON_Array* f_array = json_value_get_array(files);
        for(i=0; i<n_files; i++){
            json_array_append_string(f_array, filename[i]);
        }
        json_object_set_value(root_object, "files", files);
        json_serialize_to_file(root_object, chkptfilename);
    }
}


void restorenow(
    const char* filename,
    checkpoint_t** data,
    const int sz_data,
    int * n_data
){
    
}

#ifdef __cplusplus
}
#endif //__cplusplus