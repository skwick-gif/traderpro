import { useState, useEffect, useRef } from 'react';

function useWebSocket(url) {
  const [data, setData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = () => {
    try {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        return;
      }

      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        console.log(`WebSocket connected to ${url}`);
        setIsConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0;
      };

      wsRef.current.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          setData(parsedData);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
          setData({ raw: event.data });
        }
      };

      wsRef.current.onclose = () => {
        console.log(`WebSocket disconnected from ${url}`);
        setIsConnected(false);
        
        // Only reconnect if we haven't exceeded max attempts
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
          console.log(`Reconnecting to ${url} in ${delay}ms...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, delay);
        } else {
          setError(`Failed to connect to ${url} after ${maxReconnectAttempts} attempts`);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error(`WebSocket error on ${url}:`, error);
        setError(`Connection error: ${url}`);
        setIsConnected(false);
      };

    } catch (error) {
      console.error(`Failed to create WebSocket connection to ${url}:`, error);
      setError(`Failed to connect: ${error.message}`);
    }
  };

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url]);

  const sendMessage = (message) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  };

  const reconnect = () => {
    reconnectAttemptsRef.current = 0;
    setError(null);
    connect();
  };

  return { 
    data, 
    isConnected, 
    error, 
    sendMessage, 
    reconnect 
  };
}

export default useWebSocket;