try:
    if nuke.NUKE_VERSION_MAJOR == 14:
        if nuke.NUKE_VERSION_MINOR == 1:
            nuke.pluginAddPath("./CPP/14.1")
            print("Initialized 14.1")
        elif nuke.NUKE_VERSION_MINOR == 2:
            #nuke.pluginAddPath("./CPP/14.2")
            print("Initialized 14.2")
        else:
            print("Unsupported minor version for 14.x")
    else:
        print("Skip init for 14.x")
except:
    pass

try:
    if nuke.NUKE_VERSION_MAJOR == 15:
        if nuke.NUKE_VERSION_MINOR == 1:
            nuke.pluginAddPath("./CPP/15.1")
            print("Initialized 15.1")
        elif nuke.NUKE_VERSION_MINOR == 2:
            nuke.pluginAddPath("./CPP/15.2")
            print("Initialized 15.2")
        else:
            print("Unsupported minor version for 15.x")
    else:
        print("Skip init for 15.x")
except:
    pass

try:
    if nuke.NUKE_VERSION_MAJOR == 16:
        if nuke.NUKE_VERSION_MINOR == 0:
            nuke.pluginAddPath("./CPP/16.0")
            print("Initialized 16.0")
        else:
            print("Unsupported minor version for 16.x")
    else:
        print("Skip init for 16.x")
except:
    pass

