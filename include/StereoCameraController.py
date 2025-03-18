import os
import PySpin
import cv2

class StereoCameraController:
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

    def start_acquisition(self):
        self.left_cam.BeginAcquisition()
        self.right_cam.BeginAcquisition()

    def stop_acquisition(self):
        self.left_cam.EndAcquisition()
        self.right_cam.EndAcquisition()

    def configure_trigger(self):
        # Configura o modo de trigger para a câmera
        self.left_cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        self.right_cam.TriggerMode.SetValue(PySpin.TriggerMode_On)

        # Define o tipo de trigger como "Software"
        self.left_cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
        self.right_cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)

        # Configura o tipo de disparo (pode ser rising ou falling edge)
        self.left_cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
        self.right_cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)

    def capture_images(self):
        left_image_result = self.left_cam.GetNextImage()
        right_image_result = self.right_cam.GetNextImage()

        # Verifica se a captura foi bem-sucedida
        if left_image_result.IsIncomplete() or right_image_result.IsIncomplete():
            raise Exception("Image capture incomplete.")
        # Convert images to BGR8 format
        self.img_left = self.left_cam.GetNextImage().GetNDArray()
        self.img_right = self.right_cam.GetNextImage().GetNDArray()
        left_image_result.Release()
        right_image_result.Release()
        return self.img_left, self.img_right

    def capture_images_with_trigger(self):
        # Dispara o trigger de software
        self.left_cam.TriggerSoftware()
        self.right_cam.TriggerSoftware()

        # Captura as imagens
        left_image_result = self.left_cam.GetNextImage()
        right_image_result = self.right_cam.GetNextImage()

        # Verifica se a captura foi bem-sucedida
        if left_image_result.IsIncomplete() or right_image_result.IsIncomplete():
            raise Exception("Image capture incomplete.")
        else:
            # Converte para o formato NDArray
            self.img_left = left_image_result.GetNDArray()
            self.img_right = right_image_result.GetNDArray()

            # Libera as imagens
            left_image_result.Release()
            right_image_result.Release()

            # Print indicando que as imagens foram capturadas com sucesso
            # print("Imagens capturadas com sucesso!")

            # Retorna as imagens capturadas
            return self.img_left, self.img_right

    def save_images(self, path, counter, img_format='.png'):
        os.makedirs(os.path.join(path, 'left'), exist_ok=True)
        os.makedirs(os.path.join(path, 'right'), exist_ok=True)
        try:
            cv2.imwrite(os.path.join(os.path.join(path, 'left'), 'L' + str(counter).rjust(3, '0') + img_format), self.img_left)
            cv2.imwrite(os.path.join(os.path.join(path, 'right'), 'R' + str(counter).rjust(3, '0') + img_format), self.img_right)
            print('Image {} captured successfully.'.format(counter))
        except PySpin.SpinnakerException as ex:
            print(f"Error: {ex}")
            return False
        return True

    def camera_resolution(self):
        if self.cam_list.GetSize() == 0:
            print("Nenhuma câmera detectada")
        else:
            cam = self.cam_list.GetByIndex(0)
            cam.Init()

            # Obtém resolução da câmera
            width = cam.Width.GetValue()
            height = cam.Height.GetValue()

            return width, height

    def get_images_left(self):
        return self.img_left

    def get_images_right(self):
        return self.img_right

    def cleanup(self):
        self.left_cam.DeInit()
        self.right_cam.DeInit()
        self.cam_list.Clear()
        self.system.ReleaseInstance()
