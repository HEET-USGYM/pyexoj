// ExoJCli: run the ExoJ 1.9.2 wizard (Liu et al. 2024, GPL-3.0) without a person at the screen.
//
// The ExoJ 1.9.2 jar has no published matching source and no macro mode. This driver does what a
// user does: it opens the wizard on a stack, writes the parameter values into the wizard's own
// Parameters object (the object the wizard's text fields write into), presses Run/Next for
// step 1 (spot detection), step 2 (tracking) and step 3 (event identification), waits for each
// step to finish, and saves the wizard's own result table plus each event's raw trace.
// The algorithm code is the jar's, unchanged.
//
// Call from an ImageJ macro (run under xvfb-run fiji -batch):
//   call("ExoJCli.run", "/path/to/settings.properties");
// Settings: see README.md. Keys not given keep ExoJ's defaults; every value used is written to
// <out_dir>/<name>_params_used.tsv.

import java.awt.Window;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.SwingUtilities;
import javax.swing.table.TableModel;

import ExocytosisAnalyzer.GUI.GUIWizard;
import ExocytosisAnalyzer.datasets.Parameters;
import ExocytosisAnalyzer.datasets.Secretion;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.io.RoiDecoder;
import ij.gui.Roi;
import ij.measure.Calibration;

public class ExoJCli {

	private static Properties props;
	private static PrintWriter log;

	public static String run(String settingsPath) {
		try {
			return runChecked(settingsPath);
		} catch (Throwable t) {
			String msg = "ERROR " + t;
			IJ.log(msg);
			if (log != null) { t.printStackTrace(log); log.flush(); }
			t.printStackTrace();
			return msg;
		} finally {
			if (log != null) log.close();
			if (props != null && Boolean.parseBoolean(props.getProperty("exit", "true"))) {
				System.exit(0);
			}
		}
	}

	private static String runChecked(String settingsPath) throws Exception {
		Locale.setDefault(Locale.US); // ExoJ formats numbers with the default locale
		props = new Properties();
		try (InputStreamReader r = new InputStreamReader(new FileInputStream(settingsPath), StandardCharsets.UTF_8)) {
			props.load(r);
		}
		File outDir = new File(req("out_dir"));
		outDir.mkdirs();
		String name = props.getProperty("name", new File(req("input")).getName().replaceAll("\\.tiff?$", ""));
		log = new PrintWriter(new File(outDir, name + "_log.txt"), "UTF-8");
		long t0 = System.nanoTime();
		log("settings " + settingsPath);

		// --- input stack: single channel, 2D + time ---
		ImagePlus src = IJ.openImage(req("input"));
		if (src == null) throw new IllegalStateException("cannot open " + req("input"));
		if (src.getNChannels() > 1 || src.getNSlices() > 1)
			throw new IllegalStateException("need a single-channel 2D time series, got C=" + src.getNChannels()
					+ " Z=" + src.getNSlices() + " T=" + src.getNFrames());
		Calibration cal = src.getCalibration();
		if (props.containsKey("pixel_size_um")) {
			cal.pixelWidth = cal.pixelHeight = Double.parseDouble(props.getProperty("pixel_size_um"));
			cal.setUnit("micron");
		}
		if (props.containsKey("frame_interval_s")) {
			cal.frameInterval = Double.parseDouble(props.getProperty("frame_interval_s"));
			cal.setTimeUnit("sec");
		}
		src.setCalibration(cal);
		if (props.containsKey("roi")) { // ImageJ .roi file, e.g. a cell outline
			Roi roi = RoiDecoder.open(props.getProperty("roi"));
			if (roi == null) throw new IllegalStateException("cannot open roi " + props.getProperty("roi"));
			src.setRoi(roi);
		}
		log("input " + req("input") + " " + src.getWidth() + "x" + src.getHeight() + " frames=" + src.getStackSize()
				+ " bitDepth=" + src.getBitDepth() + " px=" + cal.pixelWidth + " " + cal.getUnit()
				+ " dt=" + cal.frameInterval + " s");

		final GUIWizard[] holder = new GUIWizard[1];
		SwingUtilities.invokeAndWait(() -> holder[0] = new GUIWizard(src));
		GUIWizard wiz = holder[0];
		closeCalibrationPrompts();
		if (src.getRoi() != null) wiz.imp.setRoi(src.getRoi());
		Parameters p = (Parameters) get(wiz, "parameters");
		JButton next = (JButton) get(wiz, "nextButton");
		ThreadPoolExecutor exec = (ThreadPoolExecutor) (ExecutorService) get(wiz, "executorService");

		// --- step 1: spot detection ---
		p.pixelSize = cal.pixelWidth;
		p.timeInterval = cal.frameInterval;
		p.framerate = 1.0 / cal.frameInterval;
		p.minSize = clamp(getInt("min_size", p.minSize), 1, 25);
		p.maxSize = clamp(getInt("max_size", p.maxSize), 1, 25);
		p.setScaleList(p.minSize, p.maxSize); // what the size fields' key listeners do
		p.SNR_Vesicle = getD("snr_vesicle", p.SNR_Vesicle);
		p.WaveletFilter = getB("wavelet", p.WaveletFilter);
		if (props.containsKey("theoretical_resolution")) {
			p.useTheoreticalResolution = true;
			p.TheoreticalResolution = getD("theoretical_resolution", p.TheoreticalResolution);
		}
		final Object vw = get(wiz, "vesicleWindows");
		onEdt(() -> {
			try {
				((JCheckBox) get(vw, "WaveletFilter")).setSelected(p.WaveletFilter);
				((JCheckBox) get(vw, "lowpass")).setSelected(false);
				vw.getClass().getMethod("updateFields").invoke(vw);
			} catch (Exception e) { throw new RuntimeException(e); }
		});
		if (getB("bleach_correct", false)) {
			// same as ticking "Correct Photobleaching": the checkbox's listener calls BleachingCorrection()
			onEdt(() -> {
				try {
					((JCheckBox) get(wiz, "photobleachingCorrection")).setSelected(true);
					Method m = GUIWizard.class.getDeclaredMethod("BleachingCorrection");
					m.setAccessible(true);
					m.invoke(wiz);
				} catch (Exception e) { throw new RuntimeException(e); }
			});
			log("photobleaching correction applied");
		}
		if (getB("delta_movie", false)) {
			onEdt(() -> {
				try {
					JCheckBox cb = (JCheckBox) get(wiz, "deltamovie");
					cb.setSelected(true);
					for (java.awt.event.ActionListener al : cb.getActionListeners())
						al.actionPerformed(new java.awt.event.ActionEvent(cb, 0, "delta"));
				} catch (Exception e) { throw new RuntimeException(e); }
			});
			log("dF movie on");
		}
		dumpParams(p, outDir, name, "step1");
		click(next);
		waitStep(wiz, exec, next, "trackingWindows", "step 1");
		int nSpots = 0;
		for (Object e : (List<?>) get(wiz, "elements")) nSpots += ((List<?>) get(e, "Vesicles")).size();
		log("step 1 done: spots=" + nSpots + " (" + secs(t0) + " s)");
		if (get(wiz, "trackingWindows") == null) return finish(wiz, outDir, name, null, t0, "no spots detected");

		// --- step 2: tracking ---
		p.SpatialsearchRadius = getInt("search_radius", p.SpatialsearchRadius);
		p.TemporalSearchDepth = getInt("temporal_depth", p.TemporalSearchDepth);
		p.minimalEventSize = getInt("min_event_size", p.minimalEventSize);
		p.showlist = false;
		final Object tw = get(wiz, "trackingWindows");
		onEdt(() -> invoke(tw, "updateFields"));
		dumpParams(p, outDir, name, "step2");
		click(next);
		waitStep(wiz, exec, next, "secretionWindows", "step 2");
		List<?> tracks = (List<?>) get(wiz, "detected_events");
		log("step 2 done: tracks=" + (tracks == null ? 0 : tracks.size()) + " (" + secs(t0) + " s)");
		if (get(wiz, "secretionWindows") == null) return finish(wiz, outDir, name, null, t0, "no tracks");

		// --- step 3: event identification ---
		p.useMinimalPointsForFitter = getB("use_min_points", p.useMinimalPointsForFitter);
		p.minimalPointsForFitter = getInt("min_points", p.minimalPointsForFitter);
		p.useExpandFrames = getB("use_expand", p.useExpandFrames);
		p.expand_frames_L = getInt("expand_pre", p.expand_frames_L);
		p.expand_frames_R = getInt("expand_post", p.expand_frames_R);
		p.fixedExpand = getB("fixed_expand", p.fixedExpand);
		p.useMinSNR = getB("use_min_snr", p.useMinSNR);
		p.min_SNR = getD("min_snr", p.min_SNR);
		p.useMaxTauFrames = getB("use_max_tau", p.useMaxTauFrames);
		p.max_tau = getD("max_tau", p.max_tau);
		p.useMinTauFrames = getB("use_min_tau", p.useMinTauFrames);
		p.min_tau = getD("min_tau", p.min_tau);
		p.useMinDecayFitterR2 = getB("use_min_decay_r2", p.useMinDecayFitterR2);
		p.min_decay_fitter_r2 = getD("min_decay_r2", p.min_decay_fitter_r2);
		p.useMaxSize = getB("use_max_size", p.useMaxSize);
		p.max_estimated_size = getD("max_est_size", p.max_estimated_size);
		p.useMinSize = getB("use_min_size", p.useMinSize);
		p.min_estimated_size = getD("min_est_size", p.min_estimated_size);
		p.useGaussienfitter2DR2 = getB("use_min_size_r2", p.useGaussienfitter2DR2);
		p.min_gaussienfitter2DR2 = getD("min_size_r2", p.min_gaussienfitter2DR2);
		p.useMaxDisplacement = getB("use_max_displacement", p.useMaxDisplacement);
		p.MaxDisplacement = getD("max_displacement", p.MaxDisplacement);
		final Object sw = get(wiz, "secretionWindows");
		onEdt(() -> {
			invoke(sw, "updateFields");
			syncBox(sw, "useMinimalPointsForFitter", p.useMinimalPointsForFitter);
			syncBox(sw, "useExpandFrames", p.useExpandFrames);
			syncBox(sw, "fixedExpand", p.fixedExpand);
			syncBox(sw, "useMaxTauFrames", p.useMaxTauFrames);
			syncBox(sw, "useMinTauFrames", p.useMinTauFrames);
			syncBox(sw, "useMaxSize", p.useMaxSize);
			syncBox(sw, "useMinSize", p.useMinSize);
			syncBox(sw, "useMaxDisplacement", p.useMaxDisplacement);
			syncBox(sw, "useMinDecayFitterR2", p.useMinDecayFitterR2);
			syncBox(sw, "useMinSNR", p.useMinSNR);
			syncBox(sw, "useGaussienfitter2DR2", p.useGaussienfitter2DR2);
		});
		dumpParams(p, outDir, name, "step3");
		click(next);
		waitStep(wiz, exec, next, "cachedAnalysisResults", "step 3");
		@SuppressWarnings("unchecked")
		List<Secretion> events = (List<Secretion>) get(wiz, "cachedAnalysisResults");
		log("step 3 done: events=" + (events == null ? 0 : events.size()) + " (" + secs(t0) + " s)");
		Parameters used = (Parameters) get(wiz, "cachedAnalysisParameters");
		if (used != null) dumpParams(used, outDir, name, "step3_snapshot");
		return finish(wiz, outDir, name, events, t0, "ok");
	}

	// ---------- output ----------

	private static String finish(GUIWizard wiz, File outDir, String name, List<Secretion> events, long t0,
			String status) throws Exception {
		if (events != null) {
			Object rw = get(wiz, "cachedResultWindow");
			if (rw == null) { // the wizard opens it on EDT after the task; give it a moment
				for (int i = 0; i < 50 && (rw = get(wiz, "cachedResultWindow")) == null; i++) Thread.sleep(200);
			}
			if (rw != null) {
				TableModel m = (TableModel) get(rw, "result_dtmS");
				try (PrintWriter w = new PrintWriter(new File(outDir, name + "_exoj_table.tsv"), "UTF-8")) {
					StringBuilder h = new StringBuilder();
					for (int c = 0; c < m.getColumnCount(); c++) h.append(c == 0 ? "" : "\t").append(clean(m.getColumnName(c)));
					w.println(h);
					for (int r = 0; r < m.getRowCount(); r++) {
						StringBuilder s = new StringBuilder();
						for (int c = 0; c < m.getColumnCount(); c++)
							s.append(c == 0 ? "" : "\t").append(clean(String.valueOf(m.getValueAt(r, c))));
						w.println(s);
					}
				}
				log("wrote ExoJ table: rows=" + m.getRowCount());
			} else log("WARNING result window not found; ExoJ table not written");
			try (PrintWriter w = new PrintWriter(new File(outDir, name + "_events.tsv"), "UTF-8");
					PrintWriter tr = new PrintWriter(new File(outDir, name + "_traces.tsv"), "UTF-8")) {
				w.println("ref\tproperty\tpeak_x\tpeak_y\tpeak_frame\tstart_frame\tfin_frame\tstart_x\tstart_y"
						+ "\tn_points\tdecay_tau_frames\tdecay_r2\tdecay_peak\tdecay_bk\tF0\tdeltaF\tSNR\tMAD"
						+ "\tmax_displacement_px\tsize2D\tsize2D_r2");
				tr.println("ref\tindex\tframe\tx\ty\tproperty\tintensity");
				for (Secretion s : events) {
					double size = Double.NaN, sizeR2 = Double.NaN;
					Parameters pp = (Parameters) get(wiz, "parameters");
					if (pp.useMaxSize || pp.useMinSize || pp.useGaussienfitter2DR2) {
						size = s.getEstimatedPeakSize2D();
						sizeR2 = s.getPeakGaussfitterRsquare2D();
					}
					w.println(s.getRef() + "\t" + s.proprety + "\t" + s.peakX + "\t" + s.peakY + "\t" + s.peakTime
							+ "\t" + s.getStartSlice() + "\t" + s.getFinSlice() + "\t" + s.getStartX() + "\t" + s.getStartY()
							+ "\t" + s.secretion_event.size() + "\t" + s.Decay_tau + "\t" + s.Decay_R2 + "\t" + s.Decay_peak
							+ "\t" + s.Decay_bk + "\t" + s.getF_zero() + "\t" + s.getDeltaF() + "\t" + s.getSNR()
							+ "\t" + s.getMAD() + "\t" + s.getMaxDisplacement() + "\t" + size + "\t" + sizeR2);
					double[] y = s.getCurve();
					for (int i = 0; i < s.secretion_event.size(); i++) {
						ExocytosisAnalyzer.datasets.Vesicle v = s.secretion_event.get(i);
						tr.println(s.getRef() + "\t" + i + "\t" + v.slice + "\t" + v.x + "\t" + v.y + "\t" + v.property
								+ "\t" + (i < y.length ? y[i] : Double.NaN));
					}
				}
			}
			log("wrote events + traces: " + events.size());
		}
		log("status " + status + " total " + secs(t0) + " s");
		// close windows so Fiji can quit
		SwingUtilities.invokeAndWait(() -> {
			for (Window w : Window.getWindows()) w.dispose();
		});
		WindowManager.closeAllWindows();
		return status;
	}

	private static void dumpParams(Parameters p, File outDir, String name, String tag) throws Exception {
		File f = new File(outDir, name + "_params_used.tsv");
		boolean header = !f.exists() || tag.equals("step1");
		try (PrintWriter w = new PrintWriter(new java.io.FileOutputStream(f, !tag.equals("step1")))) {
			if (header) w.println("stage\tfield\tvalue");
			for (Field fl : Parameters.class.getDeclaredFields()) {
				if (java.lang.reflect.Modifier.isStatic(fl.getModifiers())) continue;
				fl.setAccessible(true);
				Object v = fl.get(p);
				String s = v instanceof boolean[] ? java.util.Arrays.toString((boolean[]) v)
						: v instanceof int[] ? java.util.Arrays.toString((int[]) v) : String.valueOf(v);
				w.println(tag + "\t" + fl.getName() + "\t" + s);
			}
		}
	}

	// ---------- waiting ----------

	private static void click(JButton b) throws Exception {
		SwingUtilities.invokeAndWait(b::doClick);
	}

	// Done = the wizard's thread pool is idle, the Run/Next button is enabled again and the EDT is
	// flushed, twice in a row; or the field this step fills is set. Timeout from settings.
	private static void waitStep(GUIWizard wiz, ThreadPoolExecutor exec, JButton next, String field, String what)
			throws Exception {
		long timeout = (long) (getD("timeout_s", 3600) * 1e9);
		long start = System.nanoTime();
		int idle = 0;
		while (System.nanoTime() - start < timeout) {
			Thread.sleep(250);
			closeCalibrationPrompts();
			SwingUtilities.invokeAndWait(() -> { });
			boolean busy = exec.getActiveCount() > 0 || !exec.getQueue().isEmpty() || !next.isEnabled();
			if (!busy && get(wiz, field) != null) {
				if (++idle >= 2) return;
			} else if (!busy) {
				if (++idle >= 8) return; // finished without result (e.g. "No event detected")
			} else idle = 0;
		}
		throw new IllegalStateException(what + " timed out");
	}

	// Message dialogs (e.g. IJ.showMessage("No event detected")) would block a worker thread:
	// log their title and close them. ExoJ's progress window is left alone.
	private static void closeCalibrationPrompts() {
		for (Window w : Window.getWindows()) {
			if (w instanceof java.awt.Dialog && w.isShowing()) {
				String t = String.valueOf(((java.awt.Dialog) w).getTitle());
				if (t.startsWith("Progress")) continue;
				log("closing dialog: " + t);
				w.dispose();
			}
		}
	}

	// ---------- helpers ----------

	private static void onEdt(Runnable r) throws Exception { SwingUtilities.invokeAndWait(r); }

	private static void syncBox(Object owner, String name, boolean value) {
		try { ((JCheckBox) get(owner, name)).setSelected(value); } catch (Exception e) { throw new RuntimeException(e); }
	}

	private static void invoke(Object o, String m) {
		try { o.getClass().getMethod(m).invoke(o); } catch (Exception e) { throw new RuntimeException(e); }
	}

	static Object get(Object o, String name) throws Exception {
		Class<?> c = o.getClass();
		while (c != null) {
			try {
				Field f = c.getDeclaredField(name);
				f.setAccessible(true);
				return f.get(o);
			} catch (NoSuchFieldException e) { c = c.getSuperclass(); }
		}
		throw new NoSuchFieldException(name);
	}

	private static String req(String k) {
		String v = props.getProperty(k);
		if (v == null) throw new IllegalArgumentException("missing setting " + k);
		return v.trim();
	}
	private static int getInt(String k, int d) { return props.containsKey(k) ? Integer.parseInt(props.getProperty(k).trim()) : d; }
	private static double getD(String k, double d) { return props.containsKey(k) ? Double.parseDouble(props.getProperty(k).trim()) : d; }
	private static boolean getB(String k, boolean d) { return props.containsKey(k) ? Boolean.parseBoolean(props.getProperty(k).trim()) : d; }
	private static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }
	private static String clean(String s) { return s == null ? "" : s.replace('\t', ' ').replace('\n', ' ').trim(); }
	private static String secs(long t0) { return String.format(Locale.US, "%.1f", (System.nanoTime() - t0) / 1e9); }
	private static void log(String s) { IJ.log("[ExoJCli] " + s); if (log != null) { log.println(s); log.flush(); } }
}
