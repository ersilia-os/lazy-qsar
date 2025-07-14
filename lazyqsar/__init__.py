try:
    from .qsar import LazyBinaryQSAR
except: 
    print("You are not using the full version of lazy-qsar which has descriptord pipeline!")
    pass
from .qsar_descriptor_free import LazyBinaryQSAR as DescriptorFreeLazyQSAR

