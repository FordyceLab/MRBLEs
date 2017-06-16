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
        timepoints = range(self.sizeT)
        channels = range(self.sizeC) 
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
        timepoints = range(self.sizeT)
        channels = range(self.sizeC) 
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
        channels = range(1, c_size + 1)
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
    channels = range(channel_no)
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
    channels = range(channel_no)
    ratio_data = np.empty((data_size, channel_no))
    for ch in channels:
        # Get pixel-by-pixel ratios
        image_tmp = np.divide(images[ch, :, :], reference)
        # Get median ratio of each object
        ratio_data[:, ch] = ndi.labeled_comprehension(
            image_tmp, labels, idx, np.median, float, -1)
    return ratio_data

# DEPRECRATED
class RefSpec(object):
    """Reference Spectra
    Generate reference spectra
    """
    def __init__(self, image_files, crop = [100, 400, 100, 400], size_param = [1, 9, 10, 10, 7, 10]):
        self.image_files = image_files
        self.crop = crop
        self.ref_spec_set = None
        self.objects = None
        self.size_param = size_param

    def __close__(self):
        """Destructor of RefSpec"""
        return 0

    def readSpectra(self):
        """Read Spectra
        """
        ref_spec_set = np.array( [self.readSpectrum(file, 0,  self.size_param[idx], crop = self.crop) for idx, file in enumerate(self.image_files)] )
        self.ref_spec_set = ref_spec_set
        return ref_spec_set.T

    def readSpectrum(self, file, object_channel, size_param = [3, 9, 10, 10, 7, 10], crop = None):
        """Read Spectrum
        """
        if size_param is None:
            size_param = [3, 9, 10, 10, 7, 10]
        if crop == None: crop = self.crop
        ref_set = ImageSet(file)
        ref_set_data = ref_set.readSet()[:, crop[0]:crop[1], crop[2]:crop[3]]
        objects = self.getRefObjects(ref_set_data[object_channel], 
                                     sep_min_dist=size_param[0], min_dist=size_param[1], 
                                     param_1=size_param[2], param_2=size_param[3], 
                                     min_r=size_param[4], max_r=size_param[5])
        channels = range(ref_set_data[:, 0, 0].size)
        channels.remove(object_channel)
        ref_data = self.getRef(ref_set_data[channels])
        return ref_data

    def getRefObjects(self, object_image, sep_min_dist=3, min_dist=9, param_1=10, param_2=10, min_r=7, max_r=10):
        """Get Reference Objects
        """
        objects = Objects(object_image)
        labels, labels_annulus, circles_dim = objects.findObjects(
            sep_min_dist=sep_min_dist, min_dist=min_dist, 
            param_1=param_1, param_2=param_2, min_r=min_r, max_r=max_r)
        self.objects = labels
        return labels

    def getRef(self, image_data, back = 451):
        """Get Reference
        Get reference spectra from image set
        """
        channels = range(image_data[:, 0, 0].size)
        ref_data = np.array( [self.getMedianObjects(image_data[ch], self.objects) for ch in channels], dtype="float64" )
        ref_data = ref_data - back
        sum = ref_data.sum()
        return np.divide(ref_data, sum)

    def getMedianObjects(self, image_data, objects):
        """Get Median Objects"""
        data = ndi.median(image_data, objects)
        return data

    def getBack(image_data, square):
        """Get Background
        Get background reference of specified area
        image_data = single image used for background
        square = coordinates of region of interest [Y1, Y2, X1, X2]
        """
        c_size = image_data[:, 0, 0].size - 1
        channels = range(1, c_size + 1)
        ref_data = np.empty((c_size), dtype="float64")
        for ch in channels:
            img_tmp = image_data[ch, square[0]:square[1], square[2]:square[3]]
            ref_data[ch - 1] = np.median(img_tmp)
        sum = ref_data.sum()
        return np.divide(ref_data, sum)

def filterObjects(data, back, reference, objects_radius, back_std_factor=3, reference_std_factor=2, radius_min=3, radius_max=6):
    """Filter Objects
    Filter objects using x times SD from mean
    back = background data
    reference = reference data
    back_std_factor = x times SD from mean
    reference_std_factor = x times SD from mean
    """
    # Pre-filtered number
    pre_filter_no = data[:, 0].size

    # Mean and standard deviation of the background and the reference channel
    mean_reference = np.mean(reference)
    std_reference = np.std(reference)
    mean_back = np.mean(back)
    std_back = np.std(back)
    print(mean_reference, std_reference, mean_back, std_back)

    # Find indices of objects within search parameters
    # Check which objects are within set radius
    size_filter = np.logical_and(
        objects_radius >= radius_min, objects_radius <= radius_max)
    # Check which objects are within x SD from mean background signal
    back_filter = np.logical_and(back < (mean_back + back_std_factor * std_back),
                                 back > (mean_back - back_std_factor * std_back))
    # Check which objects are within x SD from mean reference signal
    refr_filter = np.logical_and(reference > (mean_reference - reference_std_factor * std_reference),
                                 reference < (mean_reference + reference_std_factor * std_reference))
    # Create list of indices of filtered-in objects
    filter_list = np.argwhere(np.logical_and(
        size_filter, np.logical_and(back_filter, refr_filter)))[:, 0]

    # Compare pre and post filtering object numbers
    post_filter_no = filter_list.size
    post_filter_per = int(
        ((pre_filter_no - post_filter_no) / post_filter_no) * 100)
    print("Pre-filter no:", pre_filter_no, ", Post-filter no:",
          post_filter_no, ", Filtered:", post_filter_per, "%")

    # Return list of indices of filtered-in objects
    return filter_list

class FindBeads2(object):
    """Find beads based on pure imaging.

    Attributes
    ----------
    param1 : int
        First parameters of Hough circle find algorithm.
        Defaults to PARAM1 (100).
    param2 : int
        First parameters of Hough circle find algorithm.
        Defaults to PARAM2 (5)
    """
    ## Default values
    # Default values OpenCV Hough
    global PARAM1
    PARAM1 = 200
    global PARAM2
    PARAM2 = 10
    # Default values OpenCV Thershold and Filter
    global THR_BLOCK
    THR_BLOCK = 11
    global THR_C
    THR_C = 15
    global KERNEL
    KERNEL = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (3,3))
    global FILT_ITER
    FILT_ITER = 1

    def __init__(self, bead_size, *args, **kwargs):
        self.bead_size = bead_size
        self.circles = None
        self._lbl_mask = None
        self._lbl_mask_ann = None
        self._lbl_mask_bkg = None
        self.mask_bkg_size = 15
        self.mask_bkg_buffer = 3
        self.mask_ann_size = 2
        # Default values OpenCV Hough
        self.param1 = PARAM1
        self.param2 = PARAM2
        # Default values OpenCV Thershold
        self.thr_block = THR_BLOCK
        self.thr_c = THR_C
        self.kernel = KERNEL
        self.filt_iter = FILT_ITER

    @property
    def bead_size(self):
        return self._bead_size
    @bead_size.setter
    def bead_size(self, value):
        self._bead_size = value
        self.c_min, self.c_max, self.c_min_dist = self.get_bead_dims(value)

    @property
    def bead_num(self):
        if self._lbl_mask is not None:
            return self.get_bead_num(self._lbl_mask)
        else:
            return 0

    @property
    def bead_labels(self):
        return self.get_bead_labels(self._lbl_mask)

    @staticmethod
    def get_bead_labels(mask):
        idx = np.unique(mask[mask>0])
        return idx

    @staticmethod
    def get_bead_num(mask):
        return len(np.unique(mask[mask>0]))

    @property
    def mask_bead(self):
        return self._lbl_mask+self._lbl_mask_ann

    @property
    def mask_inside(self):
        return self._lbl_mask

    @property
    def mask_outside(self):
        self._lbl_mask_bkg_incl_neg = self.lbl_mask_bkg(self._lbl_mask_incl_neg+self._lbl_mask_ann_incl_neg, 
                                                        self.mask_bkg_size, 
                                                        0)
        self._lbl_mask_bkg = self._lbl_mask_bkg_incl_neg.copy()
        self._lbl_mask_bkg[self._lbl_mask_bkg < 0] = 0
        return self._lbl_mask_bkg

    @property
    def mask_ring(self):
        return self._lbl_mask_ann

    @property
    def mask_bkg(self):
        return self._lbl_mask_bkg

    @property
    def bead_dims(self):
        props = source_properties(self._lbl_mask, self._lbl_mask)
        if not props:
            return  np.array([None, None, None]).T
        tbl = properties_table(props)
        x = tbl['xcentroid']
        y = tbl['ycentroid']
        r = tbl['equivalent_radius']
        area = tbl['area']
        dims = np.array([x,y,r]).T
        return dims

    def find(self, image):
        img = self.img2ubyte(image)
        img_thr = self.img2thr(img, self.thr_block, self.thr_c)

        labels = ndi.label(img_thr, structure=self.kernel)[0]        
        self._lbl_mask, self._lbl_mask_incl_neg = self.lbl_mask_flt(labels)

        if len(np.unique(self._lbl_mask)) <= 1:
            return

        img_thr_invert = np.invert(img_thr.copy())-254
        labels_all_bin = self._lbl_mask.copy() + img_thr_invert
        labels_all_bin[labels_all_bin > 0] = 1
        D = ndi.distance_transform_edt(labels_all_bin, sampling=3)
        labels_full = watershed(-D, markers=self._lbl_mask, mask=labels_all_bin)

        self._lbl_mask_ann, self._lbl_mask_ann_incl_neg = self.lbl_mask_flt( labels_full ) - self._lbl_mask
        self._lbl_mask_ann[self._lbl_mask_ann < 0] = 0
        self._lbl_mask[self._lbl_mask_ann_incl_neg < 0] = 0

        self._lbl_mask_bkg = self.lbl_mask_bkg(self._lbl_mask_incl_neg+self._lbl_mask_ann_incl_neg, 
                                               self.mask_bkg_size, 
                                               self.mask_bkg_buffer)
        self._lbl_mask_bkg[self._lbl_mask_bkg < 0] = 0

    @classmethod
    def lbl_mask_flt(cls, labels):
        idx = np.unique(labels)
        props = source_properties(labels, labels)
        tbl = properties_table(props)

        area_high = np.median(tbl['area'])*1.25
        area_low = np.median(tbl['area'])*0.75
        eccentricity = 0.55

        indices_ec = np.argwhere(tbl['eccentricity'] > eccentricity)
        indices_ar_max = np.argwhere(tbl['area'] > area_high)
        indices_ar_min = np.argwhere(tbl['area'] < area_low)
        indices_all = np.unique(np.concatenate((indices_ec,indices_ar_max,indices_ar_min)))
        lbl_filter = labels.copy()
        lbl_filter_incl_neg = labels.copy()
        if len(indices_all) > 0:
            for x in indices_all:
                lbl_filter[labels == idx[x+1]] = 0
                lbl_filter_incl_neg[labels == idx[x+1]] = -idx[x+1]
        return lbl_filter, lbl_filter_incl_neg

    def morph_filter(self):
        idx = cls.get_bead_labels(labels)
        props = source_properties(labels, labels)
        tbl = properties_table(props)

    @classmethod
    def lbl_mask_ann(cls, mask, size):
        mask_max = cls.mask_morph_step(size, mask)
        mask_max[mask > 0] = 0
        return mask_max

    @classmethod
    def lbl_mask_bkg(cls, mask, size, buffer=0):
        if buffer > 0:
            mask_min = cls.mask_morph_kernel(buffer, mask)
        else:
            mask_min = mask
        mask_max = cls.mask_morph_kernel(size, mask)
        mask_max[mask_min > 0] = 0
        return mask_max

    @classmethod
    def mask_morph_kernel(cls, size, mask):
        morph_mask = None
        kernel = cls.circle_kernel(abs(size))
        if size < 0:
            morph_mask = erosion(mask, kernel)
        elif size > 0:
            morph_mask = dilation(mask, kernel)
        return morph_mask

    @classmethod
    def mask_morph_step(cls, size, mask):
        morph_mask = mask.copy()
        if size < 0:
            for n in range(abs(size)):
                morph_mask = erosion(morph_mask)
        elif size > 0:
            for n in range(size):
                morph_mask = dilation(morph_mask)
        return morph_mask

    @staticmethod
    def circle_kernel(size):
        kernel = np.zeros((size, size), dtype=np.uint8)
        rr, cc = circle(np.floor(size/2), np.floor(size/2), np.ceil(size/2))
        kernel[rr, cc] = 1
        return kernel

    @staticmethod
    def create_labeled_mask(image, circles, kernel=KERNEL):
        img = image.copy()
        D = ndi.distance_transform_edt(img, sampling=3)
        markers_circles = np.zeros_like(img)
        for idx, c in enumerate(circles):
            markers_circles[int(c[1]),int(c[0])] = 1
        markers = ndi.label(markers_circles, structure=kernel)[0]
        labels = watershed(-D, markers, mask=img)
        return labels

    @staticmethod
    @accepts((np.ndarray, xd.DataArray))
    def img2ubyte(image):
        if type(image) is (xd.DataArray):
            image = image.values
        img_dtype = image.dtype
        if img_dtype is np.dtype('uint8'):
            return image
        img_min = image - image.min()
        img_max = img_min.max()
        img_conv = np.array( (img_min/img_max) * 255, dtype=np.uint8 )
        return img_conv

    @classmethod
    def img2thr(cls, image, thr_block=THR_BLOCK, thr_c=THR_C):
        img = cls.img2ubyte(image)
        img_thr = cv2.adaptiveThreshold(src = img,
                            maxValue = 1, 
                            adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            thresholdType = cv2.THRESH_BINARY,
                            blockSize = thr_block,
                            C = thr_c)
        return img_thr

    @staticmethod
    def thr2fill(image, circles, kernel=KERNEL):
        img_fill = image.copy()
        flood_mask = np.zeros((image.shape[0]+2, image.shape[1]+2), dtype='uint8')
        for idx, c in enumerate(circles):
            cv2.floodFill(image=img_fill, mask=flood_mask,
                          seedPoint = (c[1],c[0]), 
                          newVal = 2, 
                          loDiff = 0, 
                          upDiff = 1)
        img_fill[image == 0] = 0     # Add previous threshold image to filled image
        img_fill[img_fill == 0] = 1  # Set lines to 1
        img_fill[img_fill == 2] = 0  # Set background to 0
        img_fill_final = ndi.binary_fill_holes(img_fill, structure=kernel).astype(np.uint8)
        return img_fill_final

    @staticmethod
    def fill2filter(image, iter=FILT_ITER, kernel=KERNEL):
        img_filter = cv2.morphologyEx(image, 
                                      cv2.MORPH_OPEN, 
                                      kernel, 
                                      iterations = iter)
        return img_filter

    @staticmethod
    def get_bead_dims(bead_size):
        """Set default bead dimensions, min/max range, and min distance.
        """
        c_radius = bead_size / 2
        c_min = int(c_radius * 0.75)
        c_max = int(c_radius * 1.25)
        c_min_dist = (c_min * 2) - 1
        return c_min, c_max, c_min_dist

    #@classmethod
    #def img2bin(cls, image, 
    #            bead_size_param, param1=PARAM1, param2=PARAM2, 
    #            thr_block=THR_BLOCK, thr_c=THR_C, 
    #            iter=FILT_ITER, kernel=KERNEL):
    #    img = cls.img2ubyte(image)
    #    img_thr = cls.img2thr(img, thr_block, thr_c)
    #    circles = cls.circle_find(img, bead_size_param, param1, param2)
    #    img_fill = cls.thr2fill(img_thr, circles, kernel)
    #    img_final = cls.fill2filter(img_fill, iter=iter, kernel=kernel)
    #    return img_final, circles

    @classmethod
    def circle_find(cls, image, bead_size_parem, param1=PARAM1, param2=PARAM2):
        """Find circles using OpenCV Hough transform.
        """
        img = cls.img2ubyte(image)
        if type(bead_size_parem) is int:
            c_min, c_max, c_min_dist = cls.get_bead_dims(bead_size_parem)
        else:
            c_min, c_max, c_min_dist = bead_size_parem
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=c_min_dist,
                                   minRadius=c_min, 
                                   maxRadius=c_max,
                                   param1=param1,
                                   param2=param2)
        return circles[0]