try:
    from .qsar import LazyBinaryQSAR
except Exception as e: 
    print("You are not using the full version of lazy-qsar which has descriptord pipeline!")
    print(e)
    pass
from .qsar_descriptor_free import LazyBinaryQSAR as DescriptorFreeLazyQSAR


