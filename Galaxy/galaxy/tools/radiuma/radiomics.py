import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfilename, askdirectory
from PIL import Image, ImageTk
from prompt_toolkit.data_structures import Point


def radiomics_feature_generator_view(self, oid):
    def _validate(P, mi, ma):
        if P == "":
            P = mi
        if P.count(".") > 1:
            return False
        P = P.replace(".", "")
        return P.isdigit() and float(P) >= float(mi) and float(P) <= float(ma)

    def _validate_c(P, mi, ma):
        if P == "":
            P = mi
        items = P.split(",")
        for i in items:
            if i.count(".") > 1:
                return False
            j = i.replace(".", "")
            if j == '':
                j = '0'
                i = '0.'
            if not (j.isdigit() and float(i) >= float(mi) and float(i) <= float(ma)) and i != "":
                return False
        return True

    def _validate_int(P, mi, ma):
        if P == "":
            P = mi
        try:
            if P == '-':
                P = '-1'
            return int(P) >= int(mi) and int(P) <= int(ma)
        except:
            return False

    self.arr_img = ImageTk.PhotoImage(Image.open("images/arrow-icon.png").resize((20, 20)))
    main = self.newWindow
    main.none = False
    main.attributes('-toolwindow', True)
    main.title('Radiomic Feature Generator')
    main.resizable(False, False)

    self.rbg = "#FFFFFF"
    root = tk.Frame(main, bg=self.rbg)
    # menu = tk.Frame(main, bg=self.bg)
    root.grid(row=0, column=1)

    # first row
    # ttk.Separator(root, orient='horizontal').grid(column=0, row=0, columnspan=7, sticky="ew", padx=5, pady=5)
    frame = tk.Frame(root, highlightthickness=2, highlightcolor="#D1CFE2", bg=self.rbg)
    frame.grid(column=0, row=0, columnspan=6, sticky="news", padx=16, pady=(10, 5), ipady=2, ipadx=2)
    tk.Label(frame, text="Source", bg=self.rbg, fg="#1C1C28").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

    self.ra_file_folder = tk.StringVar(value=0)
    self.ra_file_folder.set(self.ras_file_folder[oid].get())

    def select_radio(clean=True):
        if self.ra_file_folder.get() == "0":
            if clean:
                if self.comp_conn[oid] == -1:
                    self.ra_file_slc_lbl.set("Select a file...")
                    self.ra_file_slc2_lbl.set("Select a file...")
                elif self.comp_conn[oid] == 0 or self.comp_conn[oid] == 2:
                    self.ra_file_slc_lbl.set("Set by connection link")
                    self.ra_file_slc2_lbl.set("Select a file...")
                else:
                    self.ra_file_slc_lbl.set("Select a file...")
                    self.ra_file_slc2_lbl.set("Set by connection link")
            self.file_select = ""
            self.file_slc_btn.config(image=self.slc_file_photo)
            self.file_slc2_btn.config(image=self.slc_file_photo)
        else:
            if clean:
                if self.comp_conn[oid] == -1:
                    self.ra_file_slc_lbl.set("Select a folder...")
                    self.ra_file_slc2_lbl.set("Select a folder...")
                elif self.comp_conn[oid] == 0 or self.comp_conn[oid] == 2:
                    self.ra_file_slc_lbl.set("Set by connection link")
                    self.ra_file_slc2_lbl.set("Select a folder...")
                else:
                    self.ra_file_slc_lbl.set("Select a folder...")
                    self.ra_file_slc2_lbl.set("Set by connection link")
            self.file_select = ""
            self.file_slc_btn.config(image=self.slc_folder_photo)
            self.file_slc2_btn.config(image=self.slc_folder_photo)

    tk.Radiobutton(frame, bg=self.rbg, text="Single file", variable=self.ra_file_folder, value=0,
                   command=select_radio).grid(column=1, row=0, padx=(0, 0))
    tk.Radiobutton(frame, bg=self.rbg, text="Folder", variable=self.ra_file_folder, value=1, command=select_radio).grid(
        column=1, row=0, padx=(150, 0))
    tk.Label(frame, text="Original image", bg=self.rbg).grid(column=0, row=2, sticky=tk.W, padx=(5, 26), pady=5)
    self.ra_file_slc_lbl = tk.StringVar(value="Select a file...")
    if self.comp_conn[oid] == 0 or self.comp_conn[oid] == 2:
        self.ra_file_slc_lbl.set("Set by connection link")
    else:
        t = self.ras_file_slc_lbl[oid].get()
        if t.startswith("Set"):
            if self.ras_file_folder[oid].get() == "0":
                t = "Select a file..."
            else:
                t = "Select a folder..."
        self.ra_file_slc_lbl.set(t)
    ttk.Entry(frame, textvariable=self.ra_file_slc_lbl, state="disabled").grid(column=0, row=2, sticky="we",
                                                                               padx=(100, 0), pady=5, ipady=10)

    def select_file():
        self.newWindow.attributes('-topmost', True)
        if self.ra_file_folder.get() == "0":
            self.file_select = askopenfilename(parent=root)
        else:
            self.file_select = askdirectory(parent=root)
        self.newWindow.attributes('-topmost', False)
        self.ra_file_slc_lbl.set(self.file_select)

    self.slc_file_photo = tk.PhotoImage(file="images/file.png")
    self.slc_folder_photo = tk.PhotoImage(file="images/folder.png")
    self.file_slc_btn = tk.Button(frame, image=self.slc_file_photo, command=select_file, bg="#9CADCE", borderwidth=0,
                                  height=40, width=40, state='disabled' if (
                    self.comp_conn[oid] == 0 or self.comp_conn[oid] == 2) else 'normal')
    self.file_slc_btn.grid(column=0, row=2, sticky="w", padx=(220, 0))

    tk.Label(frame, text="Region of interest (ROI)", bg=self.rbg).grid(column=1, row=2, sticky="e", padx=5, pady=5)
    self.ra_file_slc2_lbl = tk.StringVar(value="Select a file...")
    if self.comp_conn[oid] == 1 or self.comp_conn[oid] == 2:
        self.ra_file_slc2_lbl.set("Set by connection link")
    else:
        t = self.ras_file_slc2_lbl[oid].get()
        if t.startswith("Set"):
            if self.ras_file_folder[oid].get() == "0":
                t = "Select a file..."
            else:
                t = "Select a folder..."
        self.ra_file_slc2_lbl.set(t)
    ttk.Entry(frame, textvariable=self.ra_file_slc2_lbl, state="disabled").grid(column=2, row=2, sticky="we",
                                                                                padx=(5, 0), pady=5, columnspan=2,
                                                                                ipady=10, ipadx=4)

    def select_file2():
        self.newWindow.attributes('-topmost', True)
        if self.ra_file_folder.get() == "0":
            self.file_select2 = askopenfilename(parent=root)
        else:
            self.file_select2 = askdirectory(parent=root)
        self.newWindow.attributes('-topmost', False)
        self.ra_file_slc2_lbl.set(self.file_select2)

    self.file_slc2_btn = tk.Button(frame, image=self.slc_file_photo, command=select_file2, bg="#9CADCE", borderwidth=0,
                                   height=40, width=40, state='disabled' if (
                    self.comp_conn[oid] == 1 or self.comp_conn[oid] == 2) else 'normal')
    self.file_slc2_btn.grid(column=7, row=2, sticky="w")
    select_radio(False)

    # ttk.Separator(root, orient='horizontal').grid(column=0, row=4, columnspan=7, sticky="ew", padx=5, pady=5)
    frame = tk.Frame(root, highlightthickness=2, highlightcolor="#D1CFE2", bg=self.rbg)
    frame.grid(column=0, row=4, columnspan=6, sticky="news", padx=16, pady=5, ipady=2, ipadx=2)
    tk.Label(frame, text="Destination", bg=self.rbg, fg="#1C1C28").grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)
    self.ra_file_dest_lbl = tk.StringVar(value="Select a folder...")
    self.ra_file_dest_lbl.set(self.ras_file_dest_lbl[oid].get())
    ttk.Entry(frame, textvariable=self.ra_file_dest_lbl, state="disabled").grid(column=1, row=4, sticky="we",
                                                                                padx=(5, 0), pady=5, columnspan=2,
                                                                                ipady=10, ipadx=4)

    def select_dest_folder():
        self.newWindow.attributes('-topmost', True)
        self.dest_select = askdirectory(parent=root)
        self.newWindow.attributes('-topmost', False)
        self.ra_file_dest_lbl.set(self.dest_select)

    tk.Button(frame, image=self.slc_folder_photo, command=select_dest_folder, bg="#9CADCE", borderwidth=0, height=40,
              width=40, ).grid(column=3, row=4, sticky="w")

    # second tab
    # ttk.Separator(root, orient='horizontal').grid(column=0, row=6, columnspan=7, sticky="ew", padx=5, pady=5)
    frame = tk.Frame(root, highlightthickness=2, highlightcolor="#D1CFE2", bg=self.rbg)
    frame.grid(column=0, row=6, columnspan=6, sticky="news", padx=16, pady=5, ipady=2, ipadx=2)
    tk.Label(frame, text="Parameters", bg=self.rbg, fg="#1C1C28").grid(column=0, row=6, sticky=tk.W, padx=5, pady=5)
    tk.Label(frame, bg=self.rbg,
             text="Image modality type").grid(column=0,
                                              row=7,
                                              padx=3,
                                              pady=3, sticky="w")
    self.ra_rfg_imt_value = tk.StringVar(value="CT")
    self.ra_rfg_imt_value.set(self.ras_rfg_imt_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_imt_value, "CT", "PET", "SPECT", "MR")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=1, row=7, padx=3, pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="Intensity outlier re-segmentatin flag").grid(column=2, row=7, padx=(20, 3), pady=3, sticky="w")
    self.ra_rfg_iorf_value = tk.StringVar(value="0")
    self.ra_rfg_iorf_value.set(self.ras_rfg_iorf_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_iorf_value, "0", "1")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=3,
           row=7,
           padx=3,
           pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="Discretization type").grid(column=0,
                                              row=8,
                                              padx=3,
                                              pady=3, sticky="w")
    self.ra_rfg_dit_value = tk.StringVar(value="FBN")
    self.ra_rfg_dit_value.set(self.ras_rfg_dit_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_dit_value, "FBN", "FBS")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=1,
           row=8,
           padx=3,
           pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="Image quantization flag").grid(column=2, row=8, padx=(20, 3), pady=3, sticky="w")
    self.ra_rfg_iqf_value = tk.StringVar(value="0")
    self.ra_rfg_iqf_value.set(self.ras_rfg_iqf_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_iqf_value, "0", "1")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=3,
           row=8,
           padx=3,
           pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="Bin size/width").grid(column=0, row=9, padx=3, pady=3, sticky="w")
    self.ra_rgd_nhb_value = tk.StringVar(value="8")
    self.ra_rgd_nhb_value.set(self.ras_rgd_nhb_value[oid].get())
    ttk.Entry(frame, textvariable=self.ra_rgd_nhb_value, validate="all", justify="center",
              validatecommand=(frame.register(_validate_c), "%P", 0, 128)).grid(column=1, row=9, padx=3, pady=3,
                                                                                sticky="we")

    tk.Label(frame, bg=self.rbg, text="Re-segmentation interval range").grid(column=2, row=9, padx=(20, 2), pady=3,
                                                                             sticky="w")

    tk.Label(frame, bg=self.rbg, text="[").grid(column=3, row=9, padx=(0, 0), pady=3, sticky="w")
    self.ra_rgd_rir1_value = tk.StringVar(value=-3000)
    self.ra_rgd_rir1_value.set(self.ras_rgd_rir1_value[oid].get())
    ttk.Spinbox(frame, from_=-3000, to=0, textvariable=self.ra_rgd_rir1_value, validate="key", width=6,
                validatecommand=(frame.register(_validate_int), "%P", -3000, 0), justify="center").grid(column=3, row=9,
                                                                                                        padx=(0, 80),
                                                                                                        pady=3,
                                                                                                        sticky="e")
    tk.Label(frame, bg=self.rbg, text=",").grid(column=3, row=9, padx=(70, 0), pady=3, sticky="w")
    self.ra_rgd_rir2_value = tk.StringVar(value=3000)
    self.ra_rgd_rir2_value.set(self.ras_rgd_rir2_value[oid].get())
    ttk.Spinbox(frame, from_=0, to=3000, textvariable=self.ra_rgd_rir2_value, validate="key", width=6,
                validatecommand=(frame.register(_validate_int), "%P", 0, 3000), justify="center").grid(column=3, row=9,
                                                                                                       padx=(80, 0),
                                                                                                       pady=3,
                                                                                                       sticky="w")
    tk.Label(frame, bg=self.rbg, text="]").grid(column=3, row=9, padx=(140, 0), pady=0, sticky="w")

    tk.Label(frame, bg=self.rbg, text="Resampling (scaling) flag").grid(column=0, row=10, padx=3, pady=3, sticky="w")
    self.ra_rfg_rf_value = tk.StringVar(value="1")
    self.ra_rfg_rf_value.set(self.ras_rfg_rf_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_rf_value, "0", "1")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=1, row=10, padx=3, pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg, text="ROI partial volume threshold").grid(column=2, row=10, padx=(20, 3), pady=3,
                                                                           sticky="w")
    self.ra_rgd_roi_value = tk.StringVar(value=0.5)
    self.ra_rgd_roi_value.set(self.ras_rgd_roi_value[oid].get())
    ttk.Spinbox(frame, from_=0, to=1, increment=0.1, textvariable=self.ra_rgd_roi_value, validate="key",
                validatecommand=(frame.register(_validate), "%P", 0, 1), justify="center").grid(column=3, row=10,
                                                                                                padx=3, pady=3,
                                                                                                sticky="we")
    tk.Label(frame, bg=self.rbg, text="Image resampling interpolation type").grid(column=0, row=11, padx=3, pady=3,
                                                                                  sticky="w")
    self.ra_rfg_irit_value = tk.StringVar(value="Nearest")
    self.ra_rfg_irit_value.set(self.ras_rfg_irit_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_irit_value, "Nearest", "Linear", "Cubic")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=1, row=11, padx=3, pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg, text="Quantization type").grid(column=2, row=11, padx=(20, 3), pady=3, sticky="w")
    self.ra_rfg_qt_value = tk.StringVar(value="Uniform")
    self.ra_rfg_qt_value.set(self.ras_rfg_qt_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_qt_value, "Uniform")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=3, row=11, padx=3, pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg, text="ROI resampling interpolation type").grid(column=0, row=12, padx=3, pady=3,
                                                                                sticky="w")
    self.ra_rfg_roit_value = tk.StringVar(value="Nearest")
    self.ra_rfg_roit_value.set(self.ras_rfg_roit_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_roit_value, "Nearest", "Linear", "Cubic")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=1, row=12, padx=3, pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg, text="Intensity volume histogram (IVH) type").grid(column=2, row=12, padx=(20, 3),
                                                                                    pady=3, sticky="w")
    self.ra_rfg_ivht_value = tk.StringVar(value="1")
    self.ra_rfg_ivht_value.set(self.ras_rfg_ivht_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_ivht_value, "0", "1", "2", "3")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=3, row=12, padx=3, pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg, text="3D isotropic voxel size flag (mm)").grid(column=0, row=13, padx=3, pady=3,
                                                                                sticky="w")
    self.ra_rgd_3ivs_value = tk.StringVar(value=6.00)
    self.ra_rgd_3ivs_value.set(self.ras_rgd_3ivs_value[oid].get())
    ttk.Spinbox(frame, from_=0, to=50, increment=1, textvariable=self.ra_rgd_3ivs_value, validate="key",
                validatecommand=(frame.register(_validate), "%P", 0, 50), justify="center").grid(column=1, row=13,
                                                                                                 padx=3, pady=3,
                                                                                                 sticky="we")
    tk.Label(frame, bg=self.rbg, text="IVH discretization type").grid(column=2, row=13, padx=(20, 3), pady=3,
                                                                      sticky="w")
    self.ra_rfg_idt_value = tk.StringVar(value="0")
    self.ra_rfg_idt_value.set(self.ras_rfg_idt_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_idt_value, "0", "1")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=3, row=13, padx=3, pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg, text="2D isotropic voxel size flag (mm)").grid(column=0,
                                                                                row=14,
                                                                                padx=3,
                                                                                pady=3, sticky="w")
    self.ra_rgd_2ivs_value = tk.StringVar(value=1.00)
    self.ra_rgd_2ivs_value.set(self.ras_rgd_2ivs_value[oid].get())
    ttk.Spinbox(frame, from_=0, to=50, increment=1, textvariable=self.ra_rgd_2ivs_value, validate="key",
                validatecommand=(frame.register(_validate), "%P", 0, 50), justify="center").grid(column=1,
                                                                                                 row=14,
                                                                                                 padx=3,
                                                                                                 pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="IVH discretization binning option (number/width)").grid(column=2, row=14, padx=(20, 3), pady=3,
                                                                           sticky="w")
    self.ra_rgd_ivho_value = tk.StringVar(value=200.00)
    self.ra_rgd_ivho_value.set(self.ras_rgd_ivho_value[oid].get())
    ttk.Spinbox(frame, from_=0, to=1000, increment=1, textvariable=self.ra_rgd_ivho_value, validate="key",
                validatecommand=(frame.register(_validate), "%P", 0, 1000), justify="center").grid(column=3,
                                                                                                   row=14,
                                                                                                   padx=3,
                                                                                                   pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="Isotropic 2D voxels flag").grid(column=0,
                                                   row=15,
                                                   padx=3,
                                                   pady=3, sticky="w")
    self.ra_rfg_i2vf_value = tk.StringVar(value="0")
    self.ra_rfg_i2vf_value.set(self.ras_rfg_i2vf_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_i2vf_value, "0", "1")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=1,
           row=15,
           padx=3,
           pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="Max number of ROIs per image").grid(column=2, row=15, padx=(20, 3), pady=3, sticky="w")
    self.ra_rgd_mri_value = tk.StringVar(value="4")
    self.ra_rgd_mri_value.set(self.ras_rgd_mri_value[oid].get())
    ttk.Spinbox(frame, from_=0, to=1000000, textvariable=self.ra_rgd_mri_value, validate="key",
                validatecommand=(frame.register(_validate_int), "%P", 0, 1000000), justify="center").grid(column=3,
                                                                                                          row=15,
                                                                                                          padx=3,
                                                                                                          pady=3,
                                                                                                          sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="Round voxel intensity values flag").grid(column=0,
                                                            row=16,
                                                            padx=3,
                                                            pady=3, sticky="w")
    self.ra_rfg_rviv_value = tk.StringVar(value="0")
    self.ra_rfg_rviv_value.set(self.ras_rfg_rviv_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_rviv_value, "0", "1")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=1,
           row=16,
           padx=3,
           pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="Combine multiple ROIs to one flag").grid(column=2, row=16, padx=(20, 3), pady=3, sticky="w")
    self.ra_rfg_cmrf_value = tk.StringVar(value="0")
    self.ra_rfg_cmrf_value.set(self.ras_rfg_cmrf_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_cmrf_value, "0", "1")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=3,
           row=16,
           padx=3,
           pady=3, sticky="we")
    tk.Label(frame, bg=self.rbg,
             text="Range re-segmentation flag").grid(column=0,
                                                     row=17,
                                                     padx=3,
                                                     pady=3, sticky="w")
    self.ra_rfg_rrf_value = tk.StringVar(value="0")
    self.ra_rfg_rrf_value.set(self.ras_rfg_rrf_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_rrf_value, "0", "1")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=1,
           row=17,
           padx=3,
           pady=3, sticky="we")
    tk.Label(frame, text="Type of output data", bg=self.rbg).grid(column=2, row=17, padx=(20, 3), pady=3, sticky="w")
    self.ra_rfg_tod_value = tk.StringVar(value="2")
    self.ra_rfg_tod_value.set(self.ras_rfg_tod_value[oid].get())
    w = tk.OptionMenu(frame, self.ra_rfg_tod_value, "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12")
    w.config(indicatoron=False, compound='right', image=self.arr_img)
    w.config(highlightthickness=0, highlightbackground="black")
    w.grid(column=3,
           row=17,
           padx=3,
           pady=3, sticky="we")

    # last row
    # self.close_photo = tk.PhotoImage(file="images/close.png")
    # self.close_photo = self.close_photo.zoom(5)
    # self.close_photo = self.close_photo.subsample(3)

    def destroy():
        self.set_radiomix_params(oid)
        root.quit()
        root.destroy()
        main.none = True

    def close():
        main.quit()
        main.destroy()
        # root.quit()
        # root.destroy()

    main.protocol("WM_DELETE_WINDOW", close)
    # root.protocol("WM_DELETE_WINDOW", destroy)
    # ttk.Button(root, image=self.close_photo, command=destroy).grid(column=3, row=18, sticky="e", padx=40)
    tk.Button(root, image=self.close_photo, bg=self.rbg, command=destroy, borderwidth=0).grid(column=1, row=18,
                                                                                              columnspan=5, padx=124,
                                                                                              sticky="e", pady=(0, 5))

    # self.ok_photo = tk.PhotoImage(file="images/OK 25.png")

    # self.ok_photo = self.ok_photo.zoom(3)
    # self.ok_photo = self.ok_photo.subsample(2)
    def okf():
        for it in self.ra_rgd_nhb_value.get().split(","):
            if float(it) < 0 or float(it) > 128:
                messagebox.showerror(title="Parameter range",
                                     message="please select 'Bin size/width' paramter in [0, 128] range.")
                return
        if float(self.ra_rgd_rir1_value.get()) < -3000 or float(self.ra_rgd_rir1_value.get()) > 0:
            messagebox.showerror(title="Parameter range",
                                 message="please select 'Re-segmentation interval range' paramter in [-3000, 3000] range.")
        elif float(self.ra_rgd_rir2_value.get()) < 0 or float(self.ra_rgd_rir2_value.get()) > 3000:
            messagebox.showerror(title="Parameter range",
                                 message="please select 'Re-segmentation interval range' paramter in [-3000, 3000] range.")
        elif float(self.ra_rgd_roi_value.get()) < 0 or float(self.ra_rgd_roi_value.get()) > 1:
            messagebox.showerror(title="Parameter range",
                                 message="please select 'ROI partial volume threshold' paramter in [0, 1] range.")
        elif float(self.ra_rgd_3ivs_value.get()) < 0 or float(self.ra_rgd_3ivs_value.get()) > 50:
            messagebox.showerror(title="Parameter range",
                                 message="please select '3D isotropic voxel size flag(mm)' paramter in [0, 50] range.")
        elif float(self.ra_rgd_2ivs_value.get()) < 0 or float(self.ra_rgd_2ivs_value.get()) > 50:
            messagebox.showerror(title="Parameter range",
                                 message="please select '2D isotropic voxel size flag(mm)' paramter in [0, 50] range.")
        elif float(self.ra_rgd_ivho_value.get()) < 0 or float(self.ra_rgd_ivho_value.get()) > 1000:
            messagebox.showerror(title="Parameter range",
                                 message="please select 'IVH discretization binning option (number/width)' paramter in [0, 1000] range.")
        elif float(self.ra_rgd_mri_value.get()) < 0 or float(self.ra_rgd_mri_value.get()) > 1000:
            messagebox.showerror(title="Parameter range",
                                 message="please select 'Max no. of ROIs per image' paramter in [0, 1000] range.")
        else:
            change = False
            change = self.check_entry(self.ras_rfg_tod_value[oid], self.ra_rfg_tod_value, change)
            change = self.check_entry(self.ras_rfg_rrf_value[oid], self.ra_rfg_rrf_value, change)
            change = self.check_entry(self.ras_rfg_cmrf_value[oid], self.ra_rfg_cmrf_value, change)
            change = self.check_entry(self.ras_rfg_rviv_value[oid], self.ra_rfg_rviv_value, change)
            change = self.check_entry(self.ras_rgd_mri_value[oid], self.ra_rgd_mri_value, change)
            change = self.check_entry(self.ras_rfg_i2vf_value[oid], self.ra_rfg_i2vf_value, change)
            change = self.check_entry(self.ras_rgd_ivho_value[oid], self.ra_rgd_ivho_value, change)
            change = self.check_entry(self.ras_rgd_2ivs_value[oid], self.ra_rgd_2ivs_value, change)
            change = self.check_entry(self.ras_rfg_idt_value[oid], self.ra_rfg_idt_value, change)
            change = self.check_entry(self.ras_rgd_3ivs_value[oid], self.ra_rgd_3ivs_value, change)
            change = self.check_entry(self.ras_rfg_ivht_value[oid], self.ra_rfg_ivht_value, change)
            change = self.check_entry(self.ras_rfg_roit_value[oid], self.ra_rfg_roit_value, change)
            change = self.check_entry(self.ras_rfg_qt_value[oid], self.ra_rfg_qt_value, change)
            change = self.check_entry(self.ras_rfg_irit_value[oid], self.ra_rfg_irit_value, change)
            change = self.check_entry(self.ras_rgd_roi_value[oid], self.ra_rgd_roi_value, change)
            change = self.check_entry(self.ras_rfg_rf_value[oid], self.ra_rfg_rf_value, change)
            change = self.check_entry(self.ras_rgd_rir2_value[oid], self.ra_rgd_rir2_value, change)
            change = self.check_entry(self.ras_rgd_rir1_value[oid], self.ra_rgd_rir1_value, change)
            change = self.check_entry(self.ras_rgd_nhb_value[oid], self.ra_rgd_nhb_value, change)
            change = self.check_entry(self.ras_rfg_iqf_value[oid], self.ra_rfg_iqf_value, change)
            change = self.check_entry(self.ras_rfg_dit_value[oid], self.ra_rfg_dit_value, change)
            change = self.check_entry(self.ras_rfg_iorf_value[oid], self.ra_rfg_iorf_value, change)
            change = self.check_entry(self.ras_rfg_imt_value[oid], self.ra_rfg_imt_value, change)
            change = self.check_entry(self.ras_file_dest_lbl[oid], self.ra_file_dest_lbl, change)
            change = self.check_entry(self.ras_file_slc2_lbl[oid], self.ra_file_slc2_lbl, change)
            change = self.check_entry(self.ras_file_slc_lbl[oid], self.ra_file_slc_lbl, change)
            change = self.check_entry(self.ras_file_folder[oid], self.ra_file_folder, change)
            if change:
                self.set_object_state(oid, 'normal', 'hidden', 'hidden', 'hidden')
                self.check_connection_remove(oid)
            root.quit()
            root.destroy()

    # ttk.Button(root, image=self.ok_photo, command=okf).grid(column=3, row=18, sticky="e")
    tk.Button(root, image=self.ok_img, bg=self.rbg, command=okf, borderwidth=0).grid(column=5, row=18, sticky="e",
                                                                                     padx=(0, 16), pady=(0, 5))
    root.mainloop()
    if main.none:
        return None
    if not (self.ra_file_slc_lbl.get().startswith("Select ") or self.ra_file_slc2_lbl.get().startswith(
            "Select ") or self.ra_file_dest_lbl.get().startswith("Select ")):
        return True
    return False


class RoundedButton(tk.Canvas):
    def __init__(self, parent, width, height, corner_radius, padding, color, bg, command=None, text=""):
        tk.Canvas.__init__(self, parent, borderwidth=0,
                           relief="flat", highlightthickness=0, bg=bg)
        self.command = command

        rad = 2 * corner_radius

        def shape():
            self.create_polygon((padding, height - corner_radius - padding, padding, corner_radius + padding,
                                 padding + corner_radius, padding, width - padding - corner_radius, padding,
                                 width - padding, corner_radius + padding, width - padding,
                                 height - corner_radius - padding, width - padding - corner_radius, height - padding,
                                 padding + corner_radius, height - padding), fill=color, outline=color)
            self.create_arc((padding, padding + rad, padding + rad, padding), start=90, extent=90, fill=color,
                            outline=color)
            self.create_arc((width - padding - rad, padding, width - padding, padding + rad), start=0, extent=90,
                            fill=color, outline=color)
            self.create_arc((width - padding, height - rad - padding, width - padding - rad, height - padding),
                            start=270, extent=90, fill=color, outline=color)
            self.create_arc((padding, height - padding - rad, padding + rad, height - padding), start=180, extent=90,
                            fill=color, outline=color)
            self.create_text(width / 2, height / 2, text=text)

        id = shape()
        (x0, y0, x1, y1) = self.bbox("all")
        width = (x1 - x0)
        height = (y1 - y0)
        self.configure(width=width, height=height)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _on_press(self, event):
        self.configure(relief="sunken")

    def _on_release(self, event):
        self.configure(relief="raised")
        if self.command is not None:
            self.command()
