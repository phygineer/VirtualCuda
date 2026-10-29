// Dev mode (server started with VC_DEV=1): reload this page when a page file changes or the
// server restarts (uvicorn --reload after a .py edit). Outside dev mode the server answers 204
// once and this script goes quiet.

(() => {
  let seen = null;       // first {boot, pages} we saw
  let listening = false; // got at least one message, so errors mean "server restarting"

  function connect() {
    const es = new EventSource("/dev/livereload");
    es.onmessage = (e) => {
      listening = true;
      const { boot, pages } = JSON.parse(e.data);
      const key = `${boot}|${pages}`;
      if (seen === null) seen = key;
      else if (key !== seen) location.reload();
    };
    es.onerror = () => {
      es.close();
      if (listening) setTimeout(connect, 1000);  // server is restarting: keep trying until it's back
    };
  }
  connect();
})();
