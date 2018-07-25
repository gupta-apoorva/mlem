#ifndef _CHECKPONTING_H
#define _CHECKPONTING_H

#ifdef __cplusplus
extern "C"{
#endif //__cplusplus

typedef unsigned char byte;

typedef struct tag_checkpoint{
    unsigned long size;
    byte* stream;
}checkpoint_t;

void checkpoint_write(const char* filename, const checkpoint_t* checkpoint);
void checkpoint_read(const char* filename, checkpoint_t* checkpoint);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //_CHECKPOINTING_H