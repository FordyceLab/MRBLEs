# Depreacted functions and classes a.k.a. The Graveyard

# DEPRECTAED - BioFormats method --> Use ba.data.ImageSetRead
class ImageSetBF(object):
    """Image Set
    Load image set from file(s)
    file_path = File path(s) in list [path, path]
    """
    def __init__(self, file_path):
        """Initialize Bioformats & Java and set properties"""
        warnings.warn("Please use ImageSetRead instead!", DeprecationWarning)
        # Initiate JAVA environment and basic logging for image reader
        self.loadJVE(heap_size='8G')
        # Load image reader for images
        self._image_reader, self.metadata = self.loadImageReader(file_path)

        # Extract set information from metadata
        self.sizeC = self.getMetaDataNumber("SizeC", self.metadata)  # No. channels
        self.sizeT = self.getMetaDataNumber("SizeT", self.metadata)  # No. timepoints
        self.sizeZ = self.getMetaDataNumber("SizeZ", self.metadata)  # No. Z slices
        # TO-DO self.sizeS = self.getMetaDataNumber("SizeS", self.metadata)  # No. series/positions
        self.sizeI = self.sizeC * self.sizeT * self.sizeZ  # No. images total        
        self.imageY = self._image_reader.rdr.getSizeY()
        self.imageX = self._image_reader.rdr.getSizeX()
        self.arrayOrder = self.identifyOrder()

    def __close__(self):
        """Destructor of ImageSet"""
        self._image_reader.close()
        return 0

    @staticmethod
    def loadJVE(heap_size='2G'):
        """Load JVE
        Initiate JAVA environment and basic logging
        heap_size = Maximum size of JAVA heap, e.g. '2G' or '2024M'
        """
        jb.start_vm(class_path=bf.JARS, max_heap_size=heap_size)
        log4j.basic_config()

    @staticmethod
    def loadImageReader(file_path):
        """Initialize Bioformats reader and get metadata"""
        # Getting metadata and load image reader
        try:
            os.path.isfile(file_path)
        except IOError as io:
            print("Cannot open file:", file_path)
        except:
            print("Unexpected error:", sys.exc_info())
        else:
            image_reader = bf.ImageReader(file_path)
            metadata = bf.get_omexml_metadata(file_path)            
            return image_reader, metadata

    @staticmethod
    def getMetaDataNumber(search_keyword, metadata):
        """Extract Metadata
        Extract from bioformats metadata the number after = and between "" following the given keyword.
        search_keyword = Keyword to be searched for: "keyword"
        """
        search_string = search_keyword + r'=\"\d+\"'
        found_string = re.findall(search_string, metadata)
        if len(found_string) >= 1:
            extracted_number = int(re.findall(r'\d+', found_string[0])[0])
        else:
            extracted_number = None
        return extracted_number

    @staticmethod
    def scanPath(path, pattern=".tif"):
        """Scan Path
        Scan directory recursively for files matching the pattern.
        path = stgring, path to scan
        pattern = string, file extension
        """
        image_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if fnmatch.fnmatch(file, pattern):
                    image_files.append(os.path.join(root, file))
        return image_files

    @classmethod
    def multiScanPath(cls, paths, pattern=".tif"):
        """Multi Image Set
        Load multiple image sets from base path(s) recursively.
        paths = string, list of strings
        pattern = string, file extension
        """
        if isinstance(paths, basekeyword):
            image_files = cls.scanPath(paths, pattern=pattern)
        elif len(paths) > 1:
            image_files = map(cls.scanPath, paths, pattern=pattern)
        else:
            print("Can't resolve base path(s).")
        return image_files

    def identifyOrder(self):
        """Identify Order
        Identify the order of the image set
        """
        if self.sizeT > 1 and self.sizeC > 1:
            return "[t,c,y,x]"
        elif self.sizeT == 1 and self.sizeC > 1:
            return "[c,y,x]"
        elif self.sizeT > 1 and self.sizeC == 1:
            return "[t,y,x]"
        elif self.sizeT == 1 and self.sizeC == 1:
            return "[y,x]"
        else:
            return None

    def getIndex(self, c=0, t=0, idx=0):
        """Get Index Number
        Return index number for given channel and/or timepoint
        c = Channel number starting with 0
        t = Timepoint starting with 0
        """
        if c > 0 and c < self.sizeC or t > 0 and t < self.sizeT:
            index = c + (t * self.sizeC)
            return index
        elif c >= self.sizeC or t >= self.sizeT:
            raise IndexError
        else: return idx

    def readImage(self, idx=0, c=0, t=0, rescale=False):
        """Read Image
        Read and return single image from image set
        c = Channel number starting with 0
        t = Timepoint starting with 0
        idx = Index number starting with 0
        """
        index = self.getIndex(c=c, t=t, idx=idx)
        image = self._image_reader.read(index=index, rescale=rescale)
        return image

    def readSet(self, idx=None, c=None, t=None, rescale=False):
        """Read Set
        Read defined image set and return data array
        """
        # Set length timepoints and channels
        timepoints = xrange(self.sizeT)
        channels = xrange(self.sizeC) 
        # Iterate over timepoints and channels
        if self.arrayOrder == "[t,c,y,x]":
            image_set = np.array( [np.array( [self.readImage(c=ch, t=tp, rescale=rescale) for ch in channels] ) for tp in timepoints] )
        elif self.arrayOrder == "[c,y,x]":
            image_set = np.array( [self.readImage(c=ch, rescale=rescale) for ch in channels] )
        elif self.arrayOrder == "[t,y,x]":
            image_set = np.array( [self.readImage(t=tp, rescale=rescale) for tp in timepoints] )
        else:
            raise ValueError("Sigle image or not an image set: %s" % self.arrayOrder)
        return image_set

    def read_set_rec(self, idx=None, c=None, t=None, rescale=False):
        """Read Set
        Read defined image set and return data array
        """
        # Set length timepoints and channels
        timepoints = xrange(self.sizeT)
        channels = xrange(self.sizeC) 
        # Iterate over timepoints and channels
        if self.arrayOrder == "[t,c,y,x]":
            image_set = np.array( [np.array( [self.readImage(c=ch, t=tp, rescale=rescale) for ch in channels], dtype=[(ch, 'float64')] ) for tp in timepoints], dtype=[(tp, 'float64')] )
        elif self.arrayOrder == "[c,y,x]":
            image_set = np.array( [np.array(self.readImage(c=ch, rescale=rescale), dtype=[('ch%s'%ch, 'float64')]) for ch in channels] )
        elif self.arrayOrder == "[t,y,x]":
            image_set = np.array( [self.readImage(t=tp, rescale=rescale) for tp in timepoints], dtype=[('t')] )
        else:
            raise ValueError("Sigle image or not an image set: %s" % self.arrayOrder)
        return image_set


# DEPRECATED --> Use ba.general.FindBeads
class Objects(object):
    """Objects
    Identify objects from image and store
    """
    def __init__(self, image):
        """
        Initialization after instantiation
        Set local variables
        """
        warnings.warn("Please use FindBeads instead!", DeprecationWarning)

        # Check and/or convert image to 8 bit array. This is required for
        # object search
        self.image = self.imageConvert(image)
        self.labeled_mask = None
        self.labeled_annulus_mask = None
        self.circles_dim = None

    def __close__(self):
        """Destructor of Objects"""
        return 0

    @staticmethod
    def imageConvert(image):
        """8 Bit Convert
        Checks image data type and converts if necessary to uint8 array.
        image = M x N image array
        """
        try:
            img_type = image.dtype
        except ValueError:
            print("Not a NumPy array of image: %s" % image)
        except:
            print("Unexpected error:", sys.exc_info())
        else:
            if img_type == 'uint16':
                image = np.array( ((image / 2**16) * 2**8), dtype='uint8')
        finally:
            return image

    def findObjects(self, image=None, sep_min_dist=2, min_dist=None, param_1=20, param_2=9, min_r=3, max_r=6, ring_size=2):
        """Find Objects
        Find objects in image and return data
        """
        # Check if image is set, if not use initial image
        if image == None: img = self.image
        else: img = self.imageConvert(image)
        
        # Check if min_dist is set and set to 2 x minimuj radius
        if min_dist == None: min_dist = 2 * min_r
        # Find initial circles using Hough transform and make mask
        mask = self.makeCircleMask(min_dist=min_dist, param_1=param_1, param_2=param_2, min_r=min_r, max_r=max_r)
        # Find and separate circles using watershed on initial mask
        labels = self.separateCircles(mask)
        # Find center of circle and return dimensions
        circles_dim = self.getCircleDimensions(labels)        
        
        # Create annulus mask
        labels_annulus = labels.copy()  
        for cd in circles_dim:
            cv2.circle(labels_annulus, (int(cd[0]), int(cd[1])),
                       int(cd[2] - ring_size), (0, 0, 0), -1)
        self.labeled_mask = labels
        self.labeled_annulus_mask = labels_annulus
        self.circles_dim = circles_dim
        return labels, labels_annulus, circles_dim

    def makeCircleMask(self, image = None, min_dist=None, param_1=20, param_2=9, min_r=3, max_r=6):
        """Make Circle Mask
        """
        # Check if image is set, if not use initial image
        if image == None: img = self.image
        else: img = self.imageConvert(image)

        # Check if min_dist is set and set to 2 x minimuj radius
        if min_dist == None: min_dist = 2 * min_r
        # Find initial circles using Hough transform
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=param_1, param2=param_2, minRadius=min_r, maxRadius=max_r)[0]
        # Make mask
        mask = np.zeros(img.shape, np.uint8)
        for c in circles:
            r = int(np.ceil(c[2]))
            x, y = c[0], c[1]
            # Draw circle (line width -1 fills circle)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        return mask

    def separateCircles(self, mask, sep_min_dist=2):
        """Separate Circles
        """
        # Find and separate circles using watershed on initial mask
        D = ndi.distance_transform_edt(mask)
        localMax = peak_local_max(D, indices=False, 
                                  min_distance=sep_min_dist, 
                                  exclude_border=True, 
                                  labels=mask)
        markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=mask)
        print("Number of unique segments found: {}".format(
            len(np.unique(labels)) - 1))
        return labels

    def getCircleDimensions(self, labels, image = None):
        """
        Find center of circle and return dimensions
        """
        # Check if image is set, if not use initial image
        if image == None: img = self.image
        else: img = self.imageConvert(image)

        idx = np.arange(1, len(np.unique(labels)))
        circles_dim = np.empty((len(np.unique(labels)) - 1, 3))
        for label in idx:
            # Create single object mask
            mask_detect = np.zeros(img.shape, dtype="uint8")
            mask_detect[labels == label] = 255
            # Detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask_detect.copy(), 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = max(cnts, key=cv2.contourArea)
            # Get circle dimensions
            ((x, y), r) = cv2.minEnclosingCircle(c)
            circles_dim[label - 1, 0] = x
            circles_dim[label - 1, 1] = y
            circles_dim[label - 1, 2] = r
        return circles_dim

    def overlayImage(self, dim, img=None, ring_size=0):
        """Overlay Image
        Overlay image with circles of labeled mask
        """
        # Check if image is set, if not a copy is made. Numpy array namespaces
        # are memory locators. If no copy is made the original data is
        # manipulated.
        if img is None:
            img = self.image.copy()
        for d in dim:
            if ring_size > 0:
                cv2.circle(img, (int(d[0]), int(d[1])), int(
                    d[2] - ring_size), (0, 255, 0), 1)
            cv2.circle(img, (int(d[0]), int(d[1])), int(d[2]), (0, 255, 0), 1)
        return img

    def getBack(image_data, square):
        """Get Background
        Get background reference of specified area
        image_data = single image used for background
        square = coordinates of region of interest [Y1, Y2, X1, X2]
        """
        c_size = image_data[:, 0, 0].size - 1
        channels = xrange(1, c_size + 1)
        ref_data = np.empty((c_size), dtype="float64")
        for ch in channels:
            img_tmp = image_data[ch, square[0]:square[1], square[2]:square[3]]
            ref_data[ch - 1] = np.median(img_tmp)
        sum = ref_data.sum()
        return np.divide(ref_data, sum)


def unmix(ref_data, image_data):
    """Unmix
    Unmix the spectral images to dye images, e.g., 620nm, 630nm, 650nm images to Dy, Sm and Tm nanophospohorous lanthanides using reference spectra for each dye.
    ref_data = Reference spectra for each dye channel as Numpy Array: N x M, where N are the spectral channels and M the dye channels 
    image_data = Spectral images as NumPy array: N x M x P, where N are the spectral channels and M x P the image pixels (Y x X)
    """
    # Check if inputs are NumPy arrays and check if arrays have equal channel sizes
    try:
        ref_shape = ref_data.shape
        img_shape = image_data.shape
    except IOError:
        print("Input not NumPy array")
    if ref_shape[0] != img_shape[0]:
        raise IndexError("Number of channels not equal. Ref: ", ref_shape, " Image: ", img_shape)
    c_size = image_data[:, 0, 0].size
    y_size = image_data[0, :, 0].size
    x_size = image_data[0, 0, :].size
    ref_size = ref_data[0, :].size
    img_flat = image_data.reshape(c_size, (y_size * x_size))
    unmix_flat = np.linalg.lstsq(ref_data, img_flat)[0]
    unmix_result = unmix_flat.reshape(ref_size, y_size, x_size)
    return unmix_result


def getSpectralMedianIntensities(labels, images):
    """Get Median Intensities of each object location from the given image.
    labels = Labeled mask of objects
    images = image set of spectral images
    """
    idx = np.arange(1, len(np.unique(labels)))
    data_size = len(np.unique(labels)) - 1
    channel_no = images[:, 0, 0].size
    channels = xrange(channel_no)
    medians_data = np.empty((data_size, channel_no))
    for ch in channels:
        # Get median value of each object
        medians_data[:, ch] = ndi.labeled_comprehension(
            images[ch, :, :], labels, idx, np.median, float, -1)
    return medians_data


def getRatios(labels, images, reference):
    """Get Ratios
    Get median ratio of each object.
    """
    idx = np.arange(1, len(np.unique(labels)))
    data_size = len(np.unique(labels)) - 1
    channel_no = images[:, 0, 0].size
    channels = xrange(channel_no)
    ratio_data = np.empty((data_size, channel_no))
    for ch in channels:
        # Get pixel-by-pixel ratios
        image_tmp = np.divide(images[ch, :, :], reference)
        # Get median ratio of each object
        ratio_data[:, ch] = ndi.labeled_comprehension(
            image_tmp, labels, idx, np.median, float, -1)
    return ratio_data
