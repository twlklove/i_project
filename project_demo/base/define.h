#ifndef __DEFINE_H__
#define __DEFINE_H__

#define IS_BIG() (                                            \
{                                                             \
    u8 is_big = 1;                                            \
    u32 data = 1;                                             \
    u8 *p_data = (u8*)&data;                                  \
    if (*p_data == 1) {                                       \
        is_big = 0;                                           \
    }                                                         \
    is_big;                                                   \
})

#define B2L_32(data) (                                        \
{                                                             \
    u32 data_tmp = data;                                      \
    do {                                                      \
        if (IS_BIG()) {                                       \
            break;                                            \
        }                                                     \
        data_tmp = ((((u32)(data) & 0xff000000) >> 24)  |     \
                    (((u32)(data) & 0x00ff0000) >> 8)   |     \
                    (((u32)(data) & 0x0000ff00) << 8)   |     \
                    (((u32)(data) & 0x000000ff) << 24));      \
    } while(0);                                               \
    data_tmp;                                                 \
}) 

#define B2L_16(data) (                                        \
{                                                             \
    u16 data_tmp = data;                                      \
    do {                                                      \
        if (IS_BIG()) {                                       \
            break;                                            \
        }                                                     \
        data_tmp = ((((u16)(data) & 0xff00) >> 8)  |          \
                    (((u16)(data) & 0xff) << 8));             \
    } while(0);                                               \
    data_tmp;                                                 \
}) 

#endif
