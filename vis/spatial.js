// Embedded into the exported HTML; no server or external scripts are needed.
(function () {
    "use strict";
    const plot = document.getElementById("{plot_id}");
    const frames = plot._transitionData._frames;
    if (!frames.length) return;
    const steps = frames.map(frame => Number(frame.name));
    let index = 0;
    let playing = false;
    let generation = 0;
    let timer;

    const controls = document.createElement("div");
    controls.className = "spatial-controls";
    controls.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:12px;" +
        "padding:12px 24px;font:14px system-ui;color:#182b46;background:#f3f6fa;";
    controls.innerHTML = `
        <button type="button" data-control="play">▶ Играть</button>
        <button type="button" data-control="previous" aria-label="Предыдущий шаг">◀</button>
        <button type="button" data-control="next" aria-label="Следующий шаг">▶</button>
        <label>Шаг: <input data-control="step" type="number" style="width:85px"></label>
        <label>Шаг воспроизведения:
            <input data-control="stride" type="number" min="1" step="1" style="width:75px">
        </label>
        <span data-control="status" role="status"></span>
        <input data-control="slider" type="range" min="0" step="1"
               aria-label="Шаг симуляции" style="flex:1 1 100%;width:100%">
        <small>Наведите на организм, чтобы увидеть поле зрения. Доступны сохранённые шаги.</small>`;
    plot.parentNode.insertBefore(controls, plot);
    const control = name => controls.querySelector(`[data-control="${name}"]`);
    const play = control("play");
    const step = control("step");
    const stride = control("stride");
    const slider = control("slider");
    step.min = steps[0];
    step.max = steps[steps.length - 1];
    step.step = 1;
    slider.max = frames.length - 1;
    // Use the usual capture interval, including when the final interval is shorter.
    stride.value = steps.length > 1 ? steps[1] - steps[0] : 1;

    // Draw in screen coordinates, clipped to the world subplot. This overlay
    // cannot intercept mouse events and never changes Plotly data or world state.
    const svgNS = "http://www.w3.org/2000/svg";
    const overlay = document.createElementNS(svgNS, "svg");
    overlay.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none";
    overlay.setAttribute("aria-hidden", "true");
    const clip = document.createElementNS(svgNS, "clipPath");
    clip.id = plot.id + "-perception-clip";
    const rect = document.createElementNS(svgNS, "rect");
    clip.appendChild(rect);
    overlay.appendChild(clip);
    const circle = document.createElementNS(svgNS, "ellipse");
    circle.setAttribute("clip-path", `url(#${clip.id})`);
    circle.setAttribute("fill", "rgba(43,101,173,0.12)");
    circle.setAttribute("stroke", "#2b65ad");
    circle.setAttribute("stroke-width", "2");
    circle.setAttribute("stroke-dasharray", "6 4");
    circle.style.display = "none";
    overlay.appendChild(circle);
    let hovered = null;

    function drawVision() {
        // Plotly may replace its SVG container when redrawing a frame.
        const container = plot.querySelector(".svg-container");
        if (overlay.parentNode !== container) container.appendChild(overlay);
        if (!hovered) {
            circle.style.display = "none";
            return;
        }
        const x = plot._fullLayout.xaxis;
        const y = plot._fullLayout.yaxis;
        const radius = hovered.customdata[4];
        rect.setAttribute("x", x._offset);
        rect.setAttribute("y", y._offset);
        rect.setAttribute("width", x._length);
        rect.setAttribute("height", y._length);
        circle.setAttribute("cx", x._offset + x.l2p(hovered.x));
        circle.setAttribute("cy", y._offset + y.l2p(hovered.y));
        circle.setAttribute("rx", Math.abs(x.l2p(hovered.x + radius) - x.l2p(hovered.x)));
        circle.setAttribute("ry", Math.abs(y.l2p(hovered.y + radius) - y.l2p(hovered.y)));
        circle.style.display = "";
    }

    function clearVision() {
        hovered = null;
        circle.style.display = "none";
    }

    plot.on("plotly_hover", event => {
        hovered = event.points.find(point => point.curveNumber === 1) || null;
        drawVision();
    });
    plot.on("plotly_unhover", clearVision);
    plot.on("plotly_animatingframe", clearVision);
    plot.on("plotly_afterplot", drawVision);

    function sync() {
        slider.value = index;
        step.value = steps[index];
        play.textContent = playing ? "⏸ Пауза" : "▶ Играть";
        control("status").textContent = `${steps[index]} / ${steps[steps.length - 1]}`;
        control("previous").disabled = index === 0;
        control("next").disabled = index === frames.length - 1;
        play.disabled = frames.length === 1;
    }

    function pause() {
        playing = false;
        generation += 1;
        clearTimeout(timer);
        sync();
    }

    function show(next) {
        index = next;
        clearVision();
        sync();
        return Plotly.animate(plot, [frames[index].name], {
            mode: "immediate", frame: {duration: 0, redraw: true}, transition: {duration: 0}
        });
    }

    function advance(direction) {
        const distance = Math.max(1, Math.round(Number(stride.value) || 1));
        stride.value = distance;
        const target = steps[index] + direction * distance;
        let next = index;
        do {
            next += direction;
        } while (next > 0 && next < steps.length - 1 &&
                 (direction > 0 ? steps[next] < target : steps[next] > target));
        return Math.max(0, Math.min(steps.length - 1, next));
    }

    async function tick(token) {
        if (!playing || token !== generation) return;
        if (index === frames.length - 1) {
            pause();
            return;
        }
        try {
            await show(advance(1));
        } catch (error) {
            // A manual seek can interrupt an animation already in progress.
            if (token === generation) pause();
            return;
        }
        if (playing && token === generation) timer = setTimeout(() => tick(token), 120);
    }

    play.addEventListener("click", async () => {
        if (playing) return pause();
        const token = ++generation;
        playing = true;
        sync();
        if (index === frames.length - 1) {
            try { await show(0); } catch (error) { pause(); return; }
        }
        tick(token);
    });
    for (const [name, direction] of [["previous", -1], ["next", 1]]) {
        control(name).addEventListener("click", () => {
            pause();
            show(advance(direction)).catch(() => {});
        });
    }
    slider.addEventListener("input", () => {
        const requested = Number(slider.value);
        pause();
        show(requested).catch(() => {});
    });
    step.addEventListener("change", () => {
        const requested = Number(step.value);
        pause();
        let closest = index;
        if (Number.isFinite(requested)) {
            closest = steps.reduce((best, value, candidate) =>
                Math.abs(value - requested) < Math.abs(steps[best] - requested) ? candidate : best, 0);
        }
        show(closest).catch(() => {});
    });
    stride.addEventListener("change", () => {
        stride.value = Math.max(1, Math.round(Number(stride.value) || 1));
    });
    sync();
    drawVision();
})();
