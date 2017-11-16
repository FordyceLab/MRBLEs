# !/usr/bin/env python
"""Depreacted functions and classes a.k.a. The Graveyard"""


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

class FindBeadsImaging(object):
    """Find beads based on pure imaging.
    Parameters
    ----------
    bead_size : int
        Approximate width of beads (circles) in pixels.
    eccen_param : int, list of int
        Sets the maximum of eccentricity [0-1] of the beads (circles).
        Values close to 0 mean very circular, values closer to 1 mean very elliptical.
        Defaults to 0.65.
    area_param : int, list of int
        Sets the default min and max fraction for bead (circle) area.
        Set as single int (1+/-: 0.XX) value or of 2 values [0.XX, 1.XX].
        E.g. area_param=0.5 or area_param=[0.5, 1.5] filters all below 50% and above 150% of area calculated by approximate bead_size.
        Defaults to 0.5, which equals to [0.5, 1.5].
    Attributes
    ----------
    area_min : int or float
        Sets the minimum area in pixels.
    area_max : int or float
        Sets the maximum area in pixels.
    """

    def __init__(self, bead_size, eccen_param=0.65, area_param=0.5, border_clear=True):
        # Default values for filtering
        self._bead_size = bead_size
        self._eccen_param = eccen_param
        self._area_param = area_param
        self.set_area_limits(bead_size)
        self.filter_params = [self._eccen_param,
                              [self.area_min,
                               self.area_max]]
        self.filter_names = ['eccentricity', 'area']
        self.slice_types = ['up', 'outside']
        self.border_clear = border_clear
        # Default values OpenCV Thershold
        self.thr_block = 15
        self.thr_c = 11
        self.kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
        self.filt_iter = 1
        # Default values for local background
        self.mask_bkg_size = 11
        self.mask_bkg_buffer = 2

    # Parameter methods
    def set_area_limits(self, bead_size):
        """"Sets area limits dependent on given bead width (pixels).
        Sets: maximum and minimum area.
        """
        # Set limits
        radius = ceil(self._bead_size / 2)
        area_avg = pi * radius**2
        self.area_min, self.area_max = self.min_max(area_avg, self._area_param)

    # Main method
    # TODO: Split inside filter and whole bead filter, or change method.
    def find(self, image, circle_size=None):
        """Find objects in given image.
        """
        # Convert image to uint8
        if circle_size is None:
            img = self.img2ubyte(image)
            self._mask_radius = 0
        else:
            img, roi_mask, self._mask_radius = self.circle_roi(
                image, circle_size)
        self._masked_img = img.copy()
        # Threshold to binary image
        img_thr = self.img2thr(img, self.thr_block, self.thr_c)
        self._img_thr = img_thr
        # Label all separate parts
        mask_inside = ndi.label(img_thr, structure=self.kernel)[0]

        filter_params_inside = [[0.1 * self._bead_size **
                                 2 * np.pi, 2 * self._bead_size**2 * np.pi]]
        filter_names_inside = ['area']
        slice_types_inside = ['outside']
        self._mask_inside, self._mask_inside_neg = self.filter_mask(mask_inside,
                                                                    filter_params_inside,
                                                                    filter_names_inside,
                                                                    slice_types_inside,
                                                                    border_clear=False)
        # Check if image not empty
        if np.unique(self._mask_inside).size <= 1:
            blank_img = np.zeros_like(img)
            self._mask_bead = blank_img
            self._mask_ring = blank_img
            self._mask_outside = blank_img
            self._mask_bkg = blank_img
            return False
        # Find full bead
        img_thr_invert = (~img_thr.astype(bool)).astype(int)
        mask_all_bin = self._mask_inside + img_thr_invert
        mask_all_bin[mask_all_bin > 0] = 1
        D = ndi.distance_transform_edt(mask_all_bin, sampling=3)
        mask_full = watershed(-D, markers=self._mask_inside, mask=mask_all_bin)
        self._mask_bead, self._mask_bead_neg = self.filter_mask(mask_full,
                                                                self.filter_params,
                                                                self.filter_names,
                                                                self.slice_types,
                                                                self.border_clear)
        # Create and update final masks
        self._mask_ring = self._mask_bead - self._mask_inside
        self._mask_ring[self._mask_ring < 0] = 0
        self._mask_inside[self._mask_bead_neg < 0] = 0
        # Create outside and buffered background areas around bead
        self._mask_outside = self.make_mask_outside(
            self._mask_bead, self.mask_bkg_size, buffer=0)
        self._mask_bkg = self.make_mask_outside(
            self._mask_bead_neg, self.mask_bkg_size, buffer=self.mask_bkg_buffer)
        if circle_size is not None:
            self._mask_bkg[~roi_mask] = 0
        return True

    @staticmethod
    def img_invert(img_thr):
        """Set docstring here.
        Parameters
        ----------
        img_thr : NumPy array
            Boolean image in NumPy format.
        Returns
        -------
        img_inv : Numpy array
            Inverted boolean of the image array.
        """
        img_inv = (~img_thr.astype(bool)).astype(int)
        return img_inv

    # Properties - Settings
    @property
    def bead_size(self):
        """Get or set approximate width of beads (circles) in pixels.
        """
        return self._bead_size

    @bead_size.setter
    def bead_size(self, bead_size):
        self._bead_size = bead_size
        self.set_area_limits(bead_size)

    @property
    def area_param(self):
        """Get or set approximate width of beads (circles) in pixels.
        """
        return self._area_param

    @area_param.setter
    def area_param(self, value):
        self._area_param = value
        self.set_area_limits(self.bead_size)
        self.filter_params = [self._eccen_param,
                              [self.area_min, self.area_max]]

    @property
    def eccen_param(self):
        """Get or set approximate width of beads (circles) in pixels.
        """
        return self._eccen_param

    @area_param.setter
    def eccen_param(self, value):
        self._eccen_param = value
        self.filter_params = [self._eccen_param,
                              [self.area_min, self.area_max]]

    # Properties - Output masks
    @property
    def mask_bead(self):
        return self._mask_bead

    @property
    def mask_ring(self):
        return self._mask_ring

    @property
    def mask_inside(self):
        return self._mask_inside

    @property
    def mask_outside(self):
        return self._mask_outside

    @property
    def mask_bkg(self):
        return self._mask_bkg

    # Properties - Output values
    @property
    def bead_num(self):
        return self.get_unique_count(self._mask_bead)

    @property
    def bead_labels(self):
        return self.get_unique_values(self._mask_bead)

    @property
    def bead_dims_bead(self):
        return self.get_dimensions(self._mask_bead)

    @property
    def bead_dims_inside(self):
        return self.get_dimensions(self._mask_inside)

    # Class methods
    @classmethod
    def make_mask_outside(cls, mask, size, buffer=0):
        if buffer > 0:
            mask_min = cls.morph_mask_step(buffer, mask)
        else:
            mask_min = mask
        mask_outside = cls.morph_mask_step(size, mask)
        mask_outside[mask_min > 0] = 0
        return mask_outside

    @classmethod
    def filter_mask(cls, mask, filter_params, filter_names, slice_types, border_clear=False):
        # Get dimensions from the mask
        props = cls.get_dimensions(mask)
        # Get labels to be removed
        lbls_out = cls.filters(props, filter_params, filter_names, slice_types)
        # Create new masks
        mask_pos = mask.copy()
        mask_neg = mask.copy()
        # Set mask to 0 or negative label for labels outside limits.
        if lbls_out.size > 0:
            for lbl in lbls_out:
                mask_pos[mask == lbl] = 0
                mask_neg[mask == lbl] = -lbl
        if border_clear is True:
            clear_border(mask_pos, in_place=True)
            clear_border(mask_neg, bgval=-1, in_place=True)
        return mask_pos, mask_neg

    @classmethod
    def filters(cls, properties, filter_params, filter_names, slice_types):
        """Get labels of areas outside of limits.
        """
        lbls_out_tmp = [cls.filter(properties, param, name, stype) for param, name, stype in zip(
            filter_params, filter_names, slice_types)]
        lbls_out = np.unique(np.hstack(lbls_out_tmp))
        return lbls_out

    @classmethod
    def img2thr(cls, image, thr_block, thr_c):
        """Convert and adaptive threshold image.
        """
        img = cls.img2ubyte(image)
        img_thr = cv2.adaptiveThreshold(src=img,
                                        maxValue=1,
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        thresholdType=cv2.THRESH_BINARY,
                                        blockSize=thr_block,
                                        C=thr_c)
        return img_thr

    @classmethod
    def show_cirle_overlay(cls, image, dims=None, ring=None):
        """Show image with overlaid drawn circles of labeled mask.
        """
        img = cls.cirle_overlay(image, dims, ring)
        plt.imshow(img)

    @classmethod
    def show_cross_overlay(cls, image, dims):
        """Show image with overlay crosses.
        """
        img = color.gray2rgb(cls.img2ubyte(image))
        #dims = np.array(np.round(dims), dtype=np.int)
        for center_x, center_y, radius in zip(dims[:, 0], dims[:, 1], dims[:, 2]):
            line_y = slice(int(round(center_y) - round(radius)),
                           int(round(center_y) + round(radius)))
            line_x = slice(int(round(center_x) - round(radius)),
                           int(round(center_x) + round(radius)))
            img[int(round(center_y)), line_x] = (20, 20, 220)
            img[line_y, int(round(center_x))] = (20, 20, 220)
        plt.imshow(img)
        return img

    @classmethod
    def circle_roi(cls, image, circle_size=340):
        """Apply a circular image ROI.
        """
        img = cls.img2ubyte(image)
        dims = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                                dp=2, minDist=img.shape[0], param1=10, param2=7)
        if len(dims) > 1 or len(dims) == 0:
            return None
        cy, cx, radius = np.round(np.ravel(dims[0])).astype(np.int)
        mask = cls.sector_mask(img.shape, [cx, cy], circle_size)
        mask_img = img.copy()
        mask_img[~mask] = 0
        return mask_img, mask, [cx, cy, radius]

    # Static methods
    @staticmethod
    def sector_mask(shape, centre, radius):
        """Return a boolean mask for a circular ROI.
        """
        x, y = np.ogrid[:shape[0], :shape[1]]
        cx, cy = centre
        # convert cartesian --> polar coordinates
        r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
        # circular mask
        circmask = r2 <= radius * radius
        return circmask

    @staticmethod
    def get_unique_values(mask):
        """Get all unique positive values from an array.
        """
        values = np.unique(mask[mask > 0])
        if values.size == 0:
            values = None
        return values

    @staticmethod
    def get_unique_count(mask):
        """Get count of unique positive values from an array.
        """
        return np.unique(mask[mask > 0]).size

    @staticmethod
    def get_dimensions(mask):
        """Get dimensions of labeled regions in labeled mask.
        """
        properties = source_properties(mask, mask)
        if not properties:
            return None
        tbl = properties_table(properties)  # Convert to table
        lbl = np.array(tbl['min_value'], dtype=int)
        x = tbl['xcentroid']
        y = tbl['ycentroid']
        r = tbl['equivalent_radius']
        area = tbl['area']
        perimeter = tbl['perimeter']
        eccentricity = tbl['eccentricity']
        pdata = np.array([lbl.astype(int), x, y, r, area,
                          perimeter, eccentricity]).T
        dims = pd.DataFrame(data=pdata, columns=[
                            'label', 'x_centroid', 'y_centroid', 'radius', 'area', 'perimeter', 'eccentricity'])
        return dims

    @staticmethod
    @accepts((np.ndarray, xd.DataArray))
    def img2ubyte(image):
        """Convert image to ubuyte (uint8) and rescale to min/max.
        """
        if type(image) is (xd.DataArray):
            image = image.values
        img_dtype = image.dtype
        if img_dtype is np.dtype('uint8'):
            return image
        img_min = image - image.min()
        img_max = img_min.max()
        img_conv = np.array((img_min / img_max) * 255, dtype=np.uint8)
        return img_conv

    @staticmethod
    def cirle_overlay(image, dims, ring_size=None):
        """Overlay image with drawn circles of labeled mask.
        Parameters
        ----------
        image : NumPy array
            Base image.
        dims : NumPy array
            Array with dimensions of circles: np.array([radius, x_position, y_position], [...]): Shape: Nx3.
        ring_size: int
            Will print inside ring (annulus) with radius minus set value.
            Defaults to None, meaning not printing inside ring.
        """
        img = image.copy()
        for dim_idx, dim in enumerate(dims):
            if ring_size is not None:
                if type(ring) is int:
                    cv2.circle(img, (int(ring[dim_idx][0]), int(ring[dim_idx][1])), int(
                        ceil(ring[dim_idx][2])), (0, 255, 0), 1)
                else:
                    for dim_r in ring:
                        cv2.circle(img, (int(dim_r[0]), int(dim_r[1])), int(
                            ceil(dim_r[2])), (0, 255, 0), 1)
            cv2.circle(img, (int(dim[0]), int(dim[1])),
                       int(ceil(dim[2])), (0, 255, 0), 1)
        plt.imshow(img)
        return img

    @staticmethod
    def show_image_overlay(image, image_blend, alpha=0.3, cmap1='Greys_r', cmap2='jet'):
        """Overlay of 2 images using alpha blend.
        Parameters
        ----------
        image : NumPy array
            Base image.
        image_blend : NumPy arra
            Image to blend over base image.
        aplha : float
            Amount of blending. Value between 0 and 1.
            Defaults to 0.3.
        c_map1 : cmap
            Color scheme using cmap. See matplotlib for color schemes.
            Defaults to 'Greys_r', which are reversed grey values.
        """
        plt.axis('off')
        plt.imshow(image, cmap=cmap1)
        plt.imshow(image_blend, cmap=cmap2, interpolation='none', alpha=alpha)

    @staticmethod
    def morph_mask_step(size, mask):
        """Morph mask step-by-step using erosion or dilation.
        This function will erode or dilate step-by-step, in a loop, each labeled feature in labeled mask array.
        Parameters
        ----------
        size : int
            Set number of dilation (positive value, grow outward) or erosion (negative value, shrink inward) steps.
        mask : NumPy array
            Labeled mask to be dilated or eroded.
        """
        morph_mask = mask.copy()
        if size < 0:
            for n in range(abs(size)):
                morph_mask = erosion(morph_mask)
        elif size > 0:
            for n in range(size):
                morph_mask = dilation(morph_mask)
        return morph_mask

    @staticmethod
    def filter(properties, filter_param, filter_name, slice_type):
        """Get labels of beads outside/inside/up/down of propert limits.
        Parameters
        ----------
        properties : photutils table
            Table with feature properties from labeled mask.
            >>> from photutils import source_properties, properties_table
            >>> tbl = properties_table(properties)
            >>> properties = source_properties(mask, mask)
        filter_param : float, int, list
            Parameters to filter by.
            If provided a list it will filter by range, inside or outside).
            If provided a value it filter up or down that value.
        slice_type : string
            'outside' : < >
            'inside'  : >= <=
            'up'      : >
            'down'    : <
        """
        if type(filter_param) is list:
            if slice_type == 'outside':
                lbls_out = properties[(properties[filter_name] < filter_param[0]) | (
                    properties[filter_name] > filter_param[1])].label.values
            elif slice_type == 'inside':
                lbls_out = properties[(properties[filter_name] >= filter_param[0]) & (
                    properties[filter_name] <= filter_param[1])].label.values
        else:
            if slice_type == 'up':
                lbls_out = properties[properties[filter_name]
                                      > filter_param].label.values
            elif slice_type == 'down':
                lbls_out = properties[properties[filter_name]
                                      < filter_param].label.values
        return lbls_out

    @staticmethod
    def min_max(value, min_max):
        """Return min and max values from input value.
        Parameters
        ----------
        value : float, int
            Value to get min and max value from.
        min_max : float, list
            Percentage of min and max.
            If set by single value, e.g. +/- 0.25: min 75% / 125% of set value.
            If set by list, e.g. [0.75, 1.25]: min 75% / max 125% of set value.
        """
        if min_max is list:
            r_min = value * min_max[0]
            r_max = value * min_max[1]
        else:
            r_min = value * (1 - min_max)
            r_max = value * (1 + min_max)
        return r_min, r_max

    @staticmethod
    def eccentricity(a, b):
        """Return eccentricity by major axes.
        Parameters:
        a : float
            Size major axis a.
        b : float
            Size major axis b.
        """
        major = max([a, b])
        minor = min([a, b])
        return sqrt(1 - (minor**2 / major**2))