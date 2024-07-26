import PySpin
import os
import cv2

class StereoCameraControllerTrigger:
    def __init__(self, left_serial, right_serial):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.pixel_format = ''
        self.img_left = []
        self.img_right = []
        if self.cam_list.GetSize() < 2:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise Exception("At least two cameras are required for stereo vision.")
        for cam in self.cam_list:
            cam.Init()
            if cam.DeviceSerialNumber.ToString() == str(left_serial):
                self.left_cam = cam
            elif cam.DeviceSerialNumber.ToString() == str(right_serial):
                self.right_cam = cam
            else:
                cam.DeInit()

    def set_exposure_time(self, exposure_time):
        for cam in [self.left_cam, self.right_cam]:
            exposure_auto = cam.ExposureAuto.GetAccessMode()
            if exposure_auto == PySpin.RW:
                cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            cam.ExposureTime.SetValue(exposure_time)

    def set_exposure_mode(self, mode):
        for cam in [self.left_cam, self.right_cam]:
            exposure_auto = cam.ExposureAuto.GetAccessMode()
            if exposure_auto == PySpin.RW:
                cam.ExposureAuto.SetValue(mode)

    def set_gain(self, gain):
        for cam in [self.left_cam, self.right_cam]:
            gain_auto = cam.GainAuto.GetAccessMode()
            if gain_auto == PySpin.RW:
                cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            cam.Gain.SetValue(gain)

    def set_image_format(self, pixel_format):
        self.pixel_format = pixel_format
        for cam in [self.left_cam, self.right_cam]:
            if cam.PixelFormat.GetAccessMode() == PySpin.RW:
                cam.PixelFormat.SetValue(pixel_format)

    def get_serial_numbers(self):
        return self.left_cam.DeviceSerialNumber.ToString(), self.right_cam.DeviceSerialNumber.ToString()

    def get_model(self):
        return self.left_cam.DeviceModelName.ToString(), self.right_cam.DeviceModelName.ToString()

    def get_available_pixel_formats(self):
        pixel_formats = []
        for cam in [self.left_cam, self.right_cam]:
            try:
                node_pixel_format = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('PixelFormat'))
                if PySpin.IsAvailable(node_pixel_format) and PySpin.IsReadable(node_pixel_format):
                    entries = node_pixel_format.GetEntries()
                    for entry in entries:
                        entry = PySpin.CEnumEntryPtr(entry)  # Cast to CEnumEntryPtr
                        if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                            pixel_format = entry.GetSymbolic()
                            pixel_formats.append(pixel_format)
            except PySpin.SpinnakerException as ex:
                print(f"Error: {ex}")
        return pixel_formats

    def configure_trigger(self, cam):
        try:
            result = True

            # Ensure trigger mode off
            nodemap = cam.GetNodeMap()
            node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
            if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsWritable(node_trigger_mode):
                print("Unable to disable trigger mode (node retrieval).")
                return False

            node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
            if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
                print("Unable to disable trigger mode (entry retrieval).")
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

            # Select trigger source
            node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
            if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
                print("Unable to get trigger source (node retrieval).")
                return False

            node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
            if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(node_trigger_source_software):
                print("Unable to set trigger source to software (entry retrieval).")
                return False

            node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())

            # Turn trigger mode on
            node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
            if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
                print("Unable to enable trigger mode (entry retrieval).")
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())

            print("Trigger mode set to software.")

        except PySpin.SpinnakerException as ex:
            print("Error: %s" % ex)
            result = False

        return result

    def capture_image_by_trigger(self, cam):
        try:
            # Execute software trigger
            nodemap = cam.GetNodeMap()
            node_software_trigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
            if not PySpin.IsAvailable(node_software_trigger_cmd) or not PySpin.IsWritable(node_software_trigger_cmd):
                print("Unable to execute trigger.")
                return False

            node_software_trigger_cmd.Execute()
            print("Software trigger executed.")

            # Retrieve next received image
            image_result = cam.GetNextImage()
            if image_result.IsIncomplete():
                print("image result incomplete.")
                return None
            else:
                # convert image
                image_array = image_result.GetNDArray()

                if cam == self.left_cam:
                    self.img_left = image_array
                elif cam == self.right_cam:
                    self.img_right = image_array

                image_result.Release()
                return True

        except PySpin.SpinnakerException as ex:
            print("Error: %s" % ex)
            return False

    def save_image_by_trigger(self, path, counter, img_format='.png'):
        os.makedirs(os.path.join(path, 'left'), exist_ok=True)
        os.makedirs(os.path.join(path, 'right'), exist_ok=True)
        try:
            cv2.imwrite(os.path.join(os.path.join(path, 'left'), 'L' + str(counter).rjust(3, '0') + img_format),
                        self.img_left)
            cv2.imwrite(os.path.join(os.path.join(path, 'right'), 'R' + str(counter).rjust(3, '0') + img_format),
                        self.img_right)
            print('Image {} captured successfully.'.format(counter))
        except PySpin.SpinnakerException as ex:
            print(f"Error: {ex}")
            return False
        return True

    def main(self, save_path, img_format='.png'):
        try:
            # configure triggers
            if not self.configure_trigger(self.left_cam) or not self.configure_trigger(self.right_cam):
                return False

            # Acquire images
            self.left_cam.BeginAcquisition()
            self.right_cam.BeginAcquisition()

            if self.capture_image_by_trigger(self.left_cam) and self.capture_image_by_trigger(self.right_cam):
                print('images acquired')
                self.save_image_by_trigger(save_path, counter=1, img_format=img_format)

            self.left_cam.EndAcquisition()
            self.right_cam.EndAcquisition()
            self.left_cam.DeInit()
            self.right_cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print(f"Error: {ex}")

        finally:
            self.cam_list.Clear()
            self.system.ReleaseInstance()

if __name__ == '__main__':
    left_serial = 16378750
    right_serial = 16378734
    save_path = 'C:\\Users\\bianca.rosa\\PycharmProjects\\fringe_projection'
    controller = StereoCameraControllerTrigger(left_serial, right_serial)
    controller.main(save_path)
