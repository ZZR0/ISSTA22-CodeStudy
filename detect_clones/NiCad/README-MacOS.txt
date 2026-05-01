Recent versions of MacOS will not allow third party software such as NiCad to be run by default. 
MacOS's Gatekeeper security software will prevent NiCad from running with a message such as

    "'nicad6' can't be opened because it is from an unidentified developer".

To enable NiCad and its tools to be run on MacOS, you must first run the "macos-enable"
script in this directory, like so:

    ./macos-enable

Once this is done, MacOS will stop complaining about the NiCad tools.
